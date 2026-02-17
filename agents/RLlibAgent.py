import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any

import gymnasium as gym
import numpy as np
import pandas as pd
import ray
import tree
from ray import tune
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.numpy import convert_to_numpy
from ray.tune import ExperimentAnalysis, CheckpointConfig
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms import Algorithm
from utils.Logger import Logger
from utils.config import get_ag_cfg, get_run_cfg
import torch
import envs


@dataclass
class RLlibAgent:

    logger: Logger = field(default_factory=Logger)
    model: Any = None
    cfg: Dict[str, Any] = field(default_factory=get_ag_cfg)
    run_cfg: Dict[str, Any] = field(default_factory=get_run_cfg)  # TODO: something else
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

    def __post_init__(self):
        # Initialize ray with all existing resources (e.g. cpus)
        ray.init(ignore_reinit_error=True)

        # Set a common working directory for all trials, note this allows all trials to access resources
        # (e.g. image files) in the main project directory, but still ensure trials logging in different
        # directories, one for each trial.
        os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = "0"

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
            hyperparams.update(best_trial.config)

        # Possibly load a pretrained model
        if load_checkpoint:
            assert input_hyperparams_path is not None, ("Load a pretrained model requires `input_hyperparams_path` "
                                                        "to be the path of a Ray Tuner experiment directory.")
            raise NotImplementedError

        # Configure the RL algorithm.
        config = (
            get_trainable_cls(self.cfg['method']).get_default_config()
            .environment(
                env_id,
                env_config=env_kwargs
            )
            .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
            .env_runners(
                **self.cfg['ray_config']['env_runners']
            )
            .rl_module(
                model_config=DefaultModelConfig(
                    **self.cfg['ray_config']['rl_module']['model_config']
                )
            )
            .training(**hyperparams)
            .learners(
                **self.cfg['ray_config']['learners']
            )
            .debugging(
                seed=env_kwargs['seed']
            )
            .reporting(
                min_time_s_per_iteration=0,
            )
        )

        # Train the RL agent using Ray Tuner
        checkpoint_steps = 10000  # save checkpoint after a given number of environment steps
        tuner = tune.Tuner(
            tune.with_resources(
                get_trainable_cls(self.cfg['method']),
                resources=tune.PlacementGroupFactory(self.cfg['ray_config']['resources'])
            ),
            run_config=tune.RunConfig(
                checkpoint_config=CheckpointConfig(
                    checkpoint_frequency=checkpoint_steps // hyperparams['train_batch_size_per_learner'],
                    checkpoint_at_end=True,  # save checkpoint at the end of training
                ),
                stop={"num_env_steps_sampled_lifetime": self.cfg['train']['learning_steps']},
                storage_path=f"{os.path.abspath(os.curdir)}/{self.logger.log_dir}"
            ),
            param_space=config
        )

        results = tuner.fit()

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

        # Restore pretrained agent checkpoint
        analysis = ExperimentAnalysis(f"{os.path.abspath(os.curdir)}/{input_model_path}")
        checkpoint = analysis.get_last_checkpoint()
        # checkpoint = Checkpoint.from_directory(analysis.get_last_checkpoint().path.replace('000009', '000001'))

        agent = Algorithm.from_checkpoint(checkpoint)

        env = gym.make(env_id, **env_kwargs)
        env.max_steps = max_steps

        # Get environment observation without changing current environment state
        obs, _ = env.reset(options={'_state': env.unwrapped._state, 'goal_states': env.unwrapped.goal_states})
        done = truncated = False
        cumulative_reward = 0

        optimal_cost = env.unwrapped.distance_to_success()

        if optimal_cost == 0:
            self.logger.write('[Warning] The initial agent state is already a goal one!')

        # Get initial default LSTM state
        B = 1  # number of batches
        T = 1  # length of input observations time sequence
        L = 1  # number of LSTM layers
        state_in = {
            "h": np.zeros(shape=(agent.config.model_config['lstm_cell_size'],), dtype=np.float32),
            "c": np.zeros(shape=(agent.config.model_config['lstm_cell_size'],), dtype=np.float32),
        }
        state_in = tree.map_structure(
            lambda s: torch.from_numpy(s).reshape(B, L, *s.shape), state_in
        )

        # Get initial observation with a single time step (i.e. T=1)
        obs_hist = obs.reshape(B, T, *obs.shape)
        for i in range(max_steps):

            # Plan an action using a B=1 observation "batch".
            input_dict = {
                Columns.OBS: torch.from_numpy(obs_hist),
                Columns.STATE_IN: state_in
            }

            # Inference with no exploration
            rl_module_out = agent.get_module().forward_inference(input_dict)

            # Store output state as next input state
            state_in = rl_module_out[Columns.STATE_OUT]

            # For discrete action spaces used here, normally, an RLModule "only"
            # produces action logits, from which we then have to sample.
            # However, you can also write custom RLModules that output actions
            # directly, performing the sampling step already inside their
            # `forward_...()` methods.
            # logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])  # NO LSTM
            logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS][:, -1, :])
            action = logits[0].argmax()  # Deterministic

            self.logger.write(f"#{i}: {env.unwrapped.operators[int(action)]}")

            # Act and observe
            obs, reward, done, truncated, info = env.step(action)
            obs_hist = np.concatenate((obs_hist, obs.reshape(B, 1, *obs.shape)), axis=1)

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
        self.logger.write(f'Time seconds:{(datetime.now() - start).seconds}')

        metrics.to_excel(f"{self.logger.log_dir}/metrics.xlsx", index=False, float_format="%.2f")

        # Clean image files
        shutil.rmtree(self.logger.img_dir)
