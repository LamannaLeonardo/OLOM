import os
import warnings
import gymnasium as gym
import numpy as np
import torch
import yaml
from ConfigSpace import ConfigurationSpace
from ray import tune
from ray.tune import Tuner, RunConfig, TuneConfig, Checkpoint
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import envs
from easydict import EasyDict
from ding.policy import R2D2Policy
from ding.envs import DingEnvWrapper, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, online_logger, nstep_reward_enhancer, termination_checker
from ding.utils import set_pkg_seed

from utils.di_utils.drqn import MyDRQN


def evaluate(policy, env_kwargs, eval_episodes=5, max_ep_steps=100):

    episode_returns = list()

    # Instantiate the test environment
    env = gym.make('MNISTExib-v0', **env_kwargs)
    env.max_steps = max_ep_steps

    for ep in range(eval_episodes):
        # Reset the policy's hidden state before starting the episode
        policy._reset_eval()

        # Get environment observation without changing current environment state
        obs, _ = env.reset(seed=ep)
        cumulative_reward = i = 0
        for i in range(env.max_steps):
            # Convert observation to tensor and add batch dimension
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            data = {0: obs_tensor}  # 0 is the "env_id"

            # Get policy action
            with torch.no_grad():
                output = policy._forward_eval(data)
                action = output[0]['action']  # 0 is the "env_id"

            # Act and observe
            obs, reward, done, truncated, info = env.step(action)

            # Store cumulative reward
            cumulative_reward += reward

            # Check episode is finished
            if done or truncated:
                break

        # Store episode return
        episode_returns.append(cumulative_reward)

    return {'return_mean': np.mean(episode_returns)}


def make_env(config):
    return DingEnvWrapper(gym.make('MNISTExib-v0', **config))


def objective(hyperparams_config):

    from ding.framework import task  # Import within the function because of Ray Tuner serialization requirements
    import envs

    torch.set_printoptions(precision=3, threshold=10)

    with open(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/val/MNISTExib-v0.yaml') as f:
        val_env_configs = yaml.safe_load(f)

    with open(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/train/MNISTExib-v0.yaml') as f:
        train_configs = yaml.safe_load(f)



    collector_env_num = 4
    dqn_config = dict(
        exp_name=f'{tune.TuneContext.get_storage(tune).trial_fs_path}/di-engine',
        env=dict(
            # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
            # Env number respectively for collector and evaluator.
            collector_env_num=collector_env_num,
            evaluator_env_num=0,
            n_evaluator_episode=0,
            stop_value=np.inf
        ),
        policy=dict(
            # Whether to use cuda for network.
            cuda=torch.cuda.is_available(),
            priority=False,
            priority_IS_weight=False,
            burnin_step=hyperparams_config['burnin_step'],
            learn_unroll_len=2 * hyperparams_config['td_steps'],
            model=dict(
                obs_shape=40,
                action_shape=5,
                encoder_hidden_size_list=[512, 512],
            ),
            # Reward's future discount factor, aka. gamma.
            discount_factor=hyperparams_config['gamma'],
            nstep=hyperparams_config['td_steps'],  # steps in td error.
            # learn_mode config
            learn=dict(
                update_per_collect=hyperparams_config['num_epochs'],
                batch_size=hyperparams_config['batch_size'],
                learning_rate=hyperparams_config['lr'],
                # Frequency of target network update.
                target_update_freq=hyperparams_config['target_update_freq'],
                value_rescale=False,
            ),
            # collect_mode config
            collect=dict(
                # Either get "n_sample" samples or "n_episode" per collect.
                n_sample=hyperparams_config['n_sample'],
                # https://di-engine-test.readthedocs.io/en/latest/best_practice/rnn.html?highlight=unroll_len
                unroll_len=hyperparams_config['burnin_step'] + 2 * hyperparams_config['td_steps'],
                env_num=len(train_configs),
                traj_len_inf=True,  # Ensure trajectories are collected until termination (required by RNN)
            ),
            eval=dict(
                env_num=1,  # evaluation is performed on a single environment
                evaluator=dict(
                    eval_freq=0
                ),
            ),
            # command_mode config
            other=dict(
                # Epsilon greedy with decay.
                eps=dict(
                    # Decay type. Support ['exp', 'linear'].
                    type='exp',
                    start=0.95,
                    end=0.05,
                    decay=hyperparams_config['eps_decay'],
                ),
                replay_buffer=dict(replay_buffer_size=int(hyperparams_config['replay_buffer_size']), )
            ),
        ),
    )
    dqn_config = EasyDict(dqn_config)

    create_config = dict(
        env_manager=dict(type='subprocess_v2'),
        policy=dict(type='r2d2'),
    )
    create_config = EasyDict(create_config)

    cfg = compile_config(dqn_config, create_cfg=create_config, auto=True)

    val_env_return = []
    for k, env_kwargs in enumerate(train_configs):
        with task.start(async_mode=False, ctx=OnlineRLContext()):

            collector_env = SubprocessEnvManagerV2(
                env_fn=[lambda config=env_kwargs: make_env(config)
                        for _ in range(collector_env_num)],
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
            # task.use(online_logger(train_show_freq=10))
            # task.use(termination_checker(max_env_step=ag_config['train']['learning_steps'] * len(train_configs)))  # must be before CkptSaver
            task.use(termination_checker(max_env_step=ag_config['train']['learning_steps']))  # must be before CkptSaver
            # task.use(termination_checker(max_env_step=10000))  # must be before CkptSaver
            task.use(CkptSaver(policy, cfg.exp_name, train_freq=1000000, save_finish=True))
            task.run()

        # Evaluate on a validation set
        metrics = evaluate(policy, env_kwargs, eval_episodes=5, max_ep_steps=100)
        val_env_return.append(metrics['return_mean'])
        print(f"[Debug] Validation return mean in environment {k}: {metrics['return_mean']:.2f}")

    # tune.report(metrics={'evaluation': {'val_episode_return_mean': eval_avg_reward}},  # )
    tune.report(metrics={'val_episode_return_mean': np.mean(val_env_return)},  # )
                checkpoint=Checkpoint.from_directory(
                    f'{tune.TuneContext.get_storage(tune).trial_fs_path}/di-engine/ckpt'))


if __name__ == "__main__":

    # Input arguments: RL method and environment
    method = 'drqn'
    env_id = 'MNISTExib-v0'

    # Inhibit deprecation warnings
    os.environ['PYTHONWARNINGS'] = "ignore::DeprecationWarning"
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Get tuning agent configuration
    with open(f'{os.path.dirname(os.path.abspath(__file__))}/configs/agents/{method}.yaml') as f:
        ag_config = yaml.safe_load(f)

    # Create logs directory
    ROOT_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/res/{env_id}/{ag_config['method']}/tune/"
    os.makedirs(ROOT_DIR, exist_ok=True)
    RUN_DIR = f"{ROOT_DIR}/run{len(os.listdir(ROOT_DIR))}"

    # ray.init(local_mode=True)  # debugging

    # Set a common working directory for all trials, note this allows all trials to access resources (e.g. image files)
    # in the main project directory, but still ensure trials logging in different directories, one for each trial.
    os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = "0"

    # Get hyperparameters search space from configuration file
    hyperparams_space = ConfigurationSpace().from_serialized_dict(ag_config['tune']['search_space'])

    # Get default hyperparameters for a baseline tune trial
    default_hparams = ag_config['train']['hyperparameters']

    # BOHB: Robust and Efficient Hyperparameter Optimization at Scale, https://proceedings.mlr.press/v80/falkner18a.html
    # This is a search engine for effective hyperparameters sampling
    bohb_search = TuneBOHB(
        space=hyperparams_space,
        mode=ag_config['tune']['config']['mode'],
        metric=ag_config['tune']['config']['metric'],
        seed=123,
        points_to_evaluate=[default_hparams]  # ensure default hyperparameters configuration is tested
    )

    # Define the trials scheduler
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=1,
        reduction_factor=2,
        stop_last_trials=False,
    )

    training_function = tune.with_resources(
        objective,
        resources=tune.PlacementGroupFactory(ag_config['ray_config']['resources'])
    )

    # Initialize Ray Tuner
    tuner = Tuner(
        training_function,
        run_config=RunConfig(
            verbose=None,
            # name="tuner_exp",
            name=RUN_DIR.split('/')[-1],
            # storage_path=RUN_DIR,
            storage_path=ROOT_DIR,
        ),
        tune_config=TuneConfig(
            search_alg=bohb_search,
            scheduler=bohb_hyperband,
            **ag_config['tune']['config']
        )
    )

    # tuner = Tuner.restore('/app/res/MNISTExib-v0/DRQN/tune/run0', training_function)

    # Execute the hyperparams tuning
    results = tuner.fit()
