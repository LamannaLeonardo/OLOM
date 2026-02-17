import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import pandas as pd
from easydict import EasyDict
from ray.tune import ExperimentAnalysis
from utils.Logger import Logger
from utils.config import get_ag_cfg, get_run_cfg
import torch
from ditk import logging
from ding.policy import R2D2Policy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, online_logger, nstep_reward_enhancer, termination_checker
from ding.utils import set_pkg_seed

from utils.di_utils.drqn import MyDRQN


@dataclass
class DRQNAgent:

    logger: Logger = field(default_factory=Logger)
    cfg: Dict[str, Any] = field(default_factory=get_ag_cfg)
    run_cfg: Dict[str, Any] = field(default_factory=get_run_cfg)  # TODO: something else
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')


    def learn(self,
              env_id: str,
              env_kwargs: Dict[str, Any],
              input_hyperparams_path: str = None,
              load_checkpoint: bool = False,
              seed: int = 123) -> None:
        """
        Learn a RL model by acting in a single (Gymnasium) environment.
        :param env_id: environment name
        :param env_kwargs: environment parameters
        :param input_hyperparams_path: path of the Ray Tuner log directory that
        stores hyperparameters tuning results
        :param load_checkpoint: whether to load Ray Tuner checkpoint of the
        pretrained model during hyperparameters tuning
        :param seed: random seed for reproducibility
        :return:
        """

        # Set log directories
        self.logger.set_log_dir(f'res/{env_id}/{self.cfg["method"]}/train',
                                create_img_dir=False, create_models_dir=False)
        start = datetime.now()

        # Get default hyperparameters in agent yaml configuration
        hyperparams = self.cfg['train']['hyperparameters']

        # Load possibly tuned hyperparameters
        if input_hyperparams_path is not None:
            # Restore hyperparameters tuning experiment
            analysis = ExperimentAnalysis(f"{os.path.abspath(os.curdir)}/{input_hyperparams_path}")
            metric = "val_episode_return_mean"  # Consider some specific metric
            mode = "max"  # Maximize/minimize the metric
            best_trial = analysis.get_best_trial(metric=metric, mode=mode)
            # best_checkpoint = analysis.get_best_checkpoint(best_trial, metric=metric, mode=mode)
            hyperparams.update(best_trial.config)

        # Possibly load a pretrained model
        if load_checkpoint:
            assert input_hyperparams_path is not None, ("Load a pretrained model requires `input_hyperparams_path` "
                                                        "to be the path of a Ray Tuner experiment directory.")
            raise NotImplementedError

        alg_config = dict(
            exp_name=f"{self.logger.log_dir}/di-engine",
            env=dict(
                # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
                # Env number respectively for collector and evaluator.
                collector_env_num=self.cfg['train']['collector_env_num'],
                evaluator_env_num=self.cfg['train']['evaluator_env_num'],
                n_evaluator_episode=self.cfg['train']['n_evaluator_episode'],
                stop_value=np.inf,  # never stop because of reward (but rather use either `done` or `truncated`)
                # The path to save the game replay
                # replay_path='./mnistexib_drqn/video',
            ),
            policy=dict(
                # Whether to use cuda for network.
                cuda=torch.cuda.is_available(),
                priority=False,
                priority_IS_weight=False,
                burnin_step=hyperparams['burnin_step'],
                learn_unroll_len=2 * hyperparams['td_steps'],
                model=dict(
                    obs_shape=40,
                    action_shape=5,
                    encoder_hidden_size_list=hyperparams['encoder_hidden_size_list'],
                ),
                discount_factor=hyperparams['gamma'],
                nstep=hyperparams['td_steps'],  # steps in td error.
                # learn_mode config
                learn=dict(
                    update_per_collect=hyperparams['num_epochs'],
                    batch_size=hyperparams['batch_size'],
                    learning_rate=hyperparams['lr'],
                    # Frequency of target network update.
                    target_update_freq=hyperparams['target_update_freq'],
                    value_rescale=False,
                ),
                # collect_mode config
                collect=dict(
                    # Either get "n_sample" samples or "n_episode" per collect.
                    n_sample=hyperparams['n_sample'],
                    # https://di-engine-test.readthedocs.io/en/latest/best_practice/rnn.html?highlight=unroll_len
                    unroll_len=hyperparams['burnin_step'] + 2 * hyperparams['td_steps'],
                    env_num=self.cfg['train']['collector_env_num'],
                    traj_len_inf=True,  # Ensure trajectories are collected until termination (required by RNN)
                ),
                eval=dict(
                    env_num=self.cfg['train']['evaluator_env_num'],
                    evaluator=dict(
                        eval_freq=0
                    ),
                ),
                other=dict(
                    # Epsilon greedy with decay.
                    eps=dict(
                        # Decay type. Support ['exp', 'linear'].
                        type='exp',
                        start=0.95,
                        end=0.05,
                        decay=hyperparams['eps_decay'],
                    ),
                    replay_buffer=dict(replay_buffer_size=int(hyperparams['replay_buffer_size']), )
                ),
            ),
        )
        alg_config = EasyDict(alg_config)

        create_config = dict(
            env_manager=dict(type='subprocess_v2'),
            # env_manager=dict(type='base'),
            policy=dict(type='r2d2'),
        )
        create_config = EasyDict(create_config)

        cfg = compile_config(alg_config, create_cfg=create_config, auto=True)

        logging.getLogger().setLevel(logging.INFO)

        with task.start(async_mode=False, ctx=OnlineRLContext()):

            collector_env = SubprocessEnvManagerV2(
                env_fn=[lambda: DingEnvWrapper(gym.make(env_id, **env_kwargs))
                        for _ in range(cfg.env.collector_env_num)],
                cfg=cfg.env.manager
            )

            set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

            model = MyDRQN(**cfg.policy.model)
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            policy = R2D2Policy(cfg.policy, model=model)

            # task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(eps_greedy_handler(cfg))
            task.use(StepCollector(cfg, policy.collect_mode, collector_env))
            task.use(nstep_reward_enhancer(cfg))
            task.use(data_pusher(cfg, buffer_, group_by_env=True))  # group_by_env for R2D2 policy and DRQN
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            task.use(termination_checker(max_env_step=self.cfg['train']['learning_steps']))  # before CkptSaver to save the last checkpoint

            # Save every 10000 environment steps = n_sample * unroll_len
            checkpoint_steps = 10000
            steps_per_train_iter = hyperparams['n_sample'] * (hyperparams['burnin_step'] + 2 * hyperparams['td_steps'])
            task.use(CkptSaver(policy, cfg.exp_name, train_freq=int(math.ceil(checkpoint_steps / steps_per_train_iter)) * hyperparams['num_epochs']))
            # task.use(online_logger(record_train_iter=True, train_show_freq=1))
            task.run()

        # Save learning statistics
        metrics = pd.DataFrame([{
            'Learning steps': self.cfg['train']['learning_steps'],
            'Time seconds': (datetime.now() - start).seconds,
            'Number of environment objects': len(env_kwargs['_state']['objects'])
        }])
        metrics.to_excel(f"{self.logger.log_dir}/metrics.xlsx", index=False, float_format="%.2f")


    def solve(self,
              env_id: str,
              env_kwargs: Dict[str, Any],
              input_model_path: str,
              seed: int = 123,
              max_steps: int = 100) -> None:

        # Set log directories
        self.logger.set_log_dir(f'res/{env_id}/{self.cfg["method"]}/test')
        metrics = pd.DataFrame()
        start = datetime.now()

        # Load your trained model
        model_config = dict(
            obs_shape=40,
            action_shape=5,
            encoder_hidden_size_list=[512, 512],
        )
        model = MyDRQN(**model_config)  # !!! ensure model_config matches model architecture !!!

        # Initialize the policy with the model
        in_policy_config = dict(
            cuda=torch.cuda.is_available(),
            model=dict(
                obs_shape=40,
                action_shape=5,
            ),
            collect=dict(env_num=1),
            eval=dict(env_num=1),
        )
        policy_config = R2D2Policy.default_config()
        policy_config.update(in_policy_config)
        policy = R2D2Policy(policy_config, model=model)  # Ensure policy_config matches your policy's configuration

        state_dict = torch.load(f'{input_model_path}/ckpt/final.pth.tar', map_location=torch.device('cpu'))
        policy._load_state_dict_eval(state_dict)
        policy._model.eval()

        # Reset the policy's hidden state before starting the episode
        policy._reset_eval()

        # Instantiate the test environment
        env = gym.make(env_id, **env_kwargs)
        env.max_steps = max_steps

        # Get environment observation without changing current environment state
        obs, _ = env.reset(options={'_state': env.unwrapped._state, 'goal_states': env.unwrapped.goal_states})
        done = truncated = False
        cumulative_reward = 0

        optimal_cost = env.unwrapped.distance_to_success()

        if optimal_cost == 0:
            self.logger.write('[Warning] The initial agent state is already a goal one!')

        env_id = 0

        # episode run
        for i in range(max_steps):

            # convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            data = {env_id: obs_tensor}

            # Get action and next hidden state from policy output
            with torch.no_grad():
                output = policy._forward_eval(data)
                action = output[env_id]['action']

            # Act and observe
            self.logger.write(f"#{i}: {env.unwrapped.operators[int(action)]}")
            obs, reward, done, truncated, info = env.step(action)
            cumulative_reward += reward

            if (i + 1) % 10 == 0 or i == 0 or env.unwrapped.operators[int(action)].name == 'stop':
                # TODO: deal with this 'special' case where the initial state is already a goal one
                if optimal_cost == 0:
                    spl = float(done) * ((optimal_cost + 1) / max(i + 1, (optimal_cost + 1)))
                else:
                    spl = float(done) * (optimal_cost / max(i, optimal_cost))
                evaluation = {
                    'Objects': len(env.unwrapped._state['objects']),
                    'Non determinism': env.unwrapped.cfg['actuation_noise'],
                    'Cumulative reward': cumulative_reward,
                    'Success rate': float(done),
                    'Distance to success': env.unwrapped.distance_to_success(),
                    'SPL': spl,
                    'Time seconds': (datetime.now() - start).seconds,
                    'Steps': i + 1,
                    'Max steps': env.max_steps,
                    'Method': self.cfg['method'],
                    'Model path': input_model_path
                }
                metrics = pd.concat([metrics, pd.DataFrame([evaluation])], ignore_index=True)

            if env.unwrapped.operators[int(action)].name == 'stop' or done:
                break

        self.logger.write(f'================== METRICS ==================')
        self.logger.write(f'Success rate:{float(done)}')
        self.logger.write(f'Distance to success:{env.unwrapped.distance_to_success():.2}')
        self.logger.write(f'SPL:{spl:.2}')
        self.logger.write(f'Steps:{i + 1}')
        self.logger.write(f'Avg reward:{cumulative_reward / (i + 1)}')
        self.logger.write(f'Time seconds:{(datetime.now() - start).seconds}')

        metrics.to_excel(f"{self.logger.log_dir}/metrics.xlsx", index=False, float_format="%.2f")

        # Clean image files
        shutil.rmtree(self.logger.img_dir)
