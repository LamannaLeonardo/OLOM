import os
import random
import tempfile
import warnings
from datetime import datetime
from functools import partial
from typing import Tuple

import gymnasium
import numpy as np
import ray
import torch
import tree
import yaml
from ConfigSpace import ConfigurationSpace
from ray import tune
from ray.rllib.algorithms import Algorithm
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.env_runner_group import EnvRunnerGroup
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import ResultDict
from ray.tune import Tuner, RunConfig, TuneConfig, Checkpoint
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.registry import get_trainable_cls
import envs

with open(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/train/MNISTExib-v0.yaml') as f:
    ENV_CONFIGS = yaml.safe_load(f)


def _change_env(env_runner, new_task: int):
    """
    Recreate the environment by changing the env_runner env_config
    """
    if new_task < len(ENV_CONFIGS):
        print(f"[Debug] Setting new training environment: {new_task}")
        env_runner.config.environment(env_config={'_state': None})  # TODO: open issue, this is a
                                                                    #  TMP PATCH BECAUSE PREVIOUS KEYS ARE RETAINED
                                                                    #  WHEN THE NEW ENV CONFIG DOES NOT MENTION THEM
        env_runner.config.environment(env_config=ENV_CONFIGS[new_task])
        env_runner.make_env()
    else:
        print('[Debug] Not changing environment since all training environments have been processed')

class EnvTaskCallback(RLlibCallback):
    """Custom callback implementing `on_train_result()` for changing env config."""

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        result: dict,
        **kwargs,
    ) -> None:
        pass


# @ray.remote
def validate_serial(agent, env_id, num_episodes):
    avg_rewards = []
    env_steps = 0

    # Get validation environment configurations.
    with open(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/val/{env_id}.yaml') as f:
        val_env_configs = yaml.safe_load(f)

    for cfg in val_env_configs:

        # Create a new env instance with the given validation config.
        env = gymnasium.make(env_id, **cfg)
        env = env.unwrapped

        total_reward = 0.0
        for k in range(num_episodes):
            obs, _ = env.reset(options={'_state': env._state, 'goal_states': env.goal_states}, seed=k)
            done = truncated = False
            ep_reward = 0.0

            # Get initial default LSTM state
            B = 1  # number of batches
            T = 1  # length of input observations time sequence
            L = 1  # number of LSTM layers
            state_in = {
                "h": np.zeros(shape=(agent.config.model_config['lstm_cell_size'],), dtype=np.float32),
                "c": np.zeros(shape=(agent.config.model_config['lstm_cell_size'],), dtype=np.float32),
            }
            state_in = tree.map_structure(
                # see `_forward` in ray `TorchLSTMEncoder`
                lambda s: torch.from_numpy(s).reshape(B, L, *s.shape), state_in
            )

            obs_hist = obs.reshape(B, T, *obs.shape)
            while not (done or truncated):

                # Plan an action using a B=1 observation "batch".
                input_dict = {
                    Columns.OBS: torch.from_numpy(obs_hist),
                    Columns.STATE_IN: state_in
                }

                # Inference with no exploration
                rl_module_out = agent.get_module().forward_inference(input_dict)

                # Store output state as next input state
                state_in = rl_module_out[Columns.STATE_OUT]

                # logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])  # NO LSTM
                logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS][:, -1, :])
                action = logits[0].argmax()

                obs, reward, done, truncated, _ = env.step(action)

                obs_hist = np.concatenate((obs_hist, obs.reshape(B, 1, *obs.shape)), axis=1)
                ep_reward += reward
                env_steps += 1
            total_reward += ep_reward
        env_avg_reward = total_reward / num_episodes
        avg_rewards.append(env_avg_reward)
    return {'val_episode_return_mean': np.mean(avg_rewards), 'env_steps': env_steps}


def validate_serial_train(agent, train_cfg, num_episodes):
    avg_rewards = []
    env_steps = 0

    # Create a new env instance with the given validation config.
    env = gymnasium.make(env_id, **train_cfg)
    env = env.unwrapped

    total_reward = 0.0
    for k in range(num_episodes):
        obs, _ = env.reset(options={'_state': env._state, 'goal_states': env.goal_states}, seed=k)
        done = truncated = False
        ep_reward = 0.0

        # Get initial default LSTM state
        B = 1  # number of batches
        T = 1  # length of input observations time sequence
        L = 1  # number of LSTM layers
        state_in = {
            "h": np.zeros(shape=(agent.config.model_config['lstm_cell_size'],), dtype=np.float32),
            "c": np.zeros(shape=(agent.config.model_config['lstm_cell_size'],), dtype=np.float32),
        }
        state_in = tree.map_structure(
            # see `_forward` in ray `TorchLSTMEncoder`
            lambda s: torch.from_numpy(s).reshape(B, L, *s.shape), state_in
        )

        obs_hist = obs.reshape(B, T, *obs.shape)
        while not (done or truncated):

            # Plan an action using a B=1 observation "batch".
            input_dict = {
                Columns.OBS: torch.from_numpy(obs_hist),
                Columns.STATE_IN: state_in
            }

            # Inference with no exploration
            rl_module_out = agent.get_module().forward_inference(input_dict)

            # Store output state as next input state
            state_in = rl_module_out[Columns.STATE_OUT]

            # logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])  # NO LSTM
            logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS][:, -1, :])

            # action = np.random.choice(env.action_space.n, p=softmax(logits[0]))
            action = logits[0].argmax()  # Deterministic

            obs, reward, done, truncated, _ = env.step(action)

            obs_hist = np.concatenate((obs_hist, obs.reshape(B, 1, *obs.shape)), axis=1)
            ep_reward += reward
            env_steps += 1
        total_reward += ep_reward
    env_avg_reward = total_reward / num_episodes
    avg_rewards.append(env_avg_reward)
    return {'val_episode_return_mean': np.mean(avg_rewards), 'env_steps': env_steps}


# @ray.remote
def evaluate_on_configs(worker, env_configs, num_episodes):
    avg_rewards = []
    env_steps = 0
    # TODO: parallelize evaluation according to `num_envs_per_env_runner`

    for cfg in env_configs:

        # Create a new env instance with the given validation config.
        env = gymnasium.make(env_id, **cfg)
        env = env.unwrapped

        total_reward = 0.0
        for _ in range(num_episodes):
            obs, _ = env.reset(options={'_state': env._state, 'goal_states': env.goal_states})
            done = truncated = False
            ep_reward = 0.0

            # Get initial default LSTM state
            B = 1  # number of batches
            T = 1  # length of input observations time sequence
            L = 1  # number of LSTM layers
            state_in = {
                "h": np.zeros(shape=(worker.config.model_config['lstm_cell_size'],), dtype=np.float32),
                "c": np.zeros(shape=(worker.config.model_config['lstm_cell_size'],), dtype=np.float32),
            }
            state_in = tree.map_structure(
                # see `_forward` in ray `TorchLSTMEncoder`
                lambda s: torch.from_numpy(s).reshape(B, L, *s.shape), state_in
            )

            obs_hist = obs.reshape(B, T, *obs.shape)
            while not (done or truncated):

                # Plan an action using a B=1 observation "batch".
                input_dict = {
                    Columns.OBS: torch.from_numpy(obs_hist),
                    Columns.STATE_IN: state_in
                }

                # Inference with no exploration
                rl_module_out = worker.module.forward_inference(input_dict)

                # Store output state as next input state
                state_in = rl_module_out[Columns.STATE_OUT]

                # logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])  # NO LSTM
                logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS][:, -1, :])
                action = logits[0].argmax()

                obs, reward, done, truncated, _ = env.step(action)

                obs_hist = np.concatenate((obs_hist, obs.reshape(B, 1, *obs.shape)), axis=1)
                ep_reward += reward
                env_steps += 1
            total_reward += ep_reward
        env_avg_reward = total_reward / num_episodes
        avg_rewards.append(env_avg_reward)

    return {'avg_rewards': avg_rewards, 'env_steps': env_steps}


def custom_evaluation_function(
        algorithm: Algorithm,
        eval_workers: EnvRunnerGroup) -> Tuple[ResultDict, int, int]:

    # Get validation environment configurations.
    with open(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/val/{env_id}.yaml') as f:
        validation_env_configs = yaml.safe_load(f)

    num_episodes = 5  # number of evaluation episodes for every environment

    num_workers = max(1, eval_workers.num_remote_workers())
    # Partition the list of env configs among the available workers.
    partitions = [validation_env_configs[v::num_workers] for v in range(num_workers)]

    # Launch evaluation concurrently on each worker.
    offset = 1 if eval_workers.num_remote_workers() > 0 else 0
    eval_results_futures = eval_workers.foreach_env_runner(
        lambda ev: evaluate_on_configs(ev, partitions[ev.worker_index - offset], num_episodes)
    )
    avg_rewards = [metrics['avg_rewards'] for metrics in eval_results_futures]
    env_steps = [metrics['env_steps'] for metrics in eval_results_futures]

    # Return the average episode return further averaged
    # over all episodes in all environments
    env_steps = sum(env_steps)
    agent_steps = env_steps  # when episode is single-agent: agent_steps = env_steps

    # Log evaluation metrics
    algorithm.metrics.set_value(('evaluation', 'val_episode_return_mean'), np.mean(avg_rewards))
    algorithm.metrics.set_value(('val_episode_return_mean'), np.mean(avg_rewards))
    return {'val_episode_return_mean': np.mean(avg_rewards)}, env_steps, agent_steps


def objective(hyperparams_config):
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get training set of environments
    with open(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/train/{env_id}.yaml') as f:
        train_env_configs = yaml.safe_load(f)

    # Configure the algorithm.
    config = (
        get_trainable_cls(ag_config['method']).get_default_config()
        .environment(
            env_id,
            env_config=train_env_configs[0]
        )
        .callbacks(EnvTaskCallback)
        .env_runners(
            **ag_config['ray_config']['env_runners']
        )
        .rl_module(
            model_config=DefaultModelConfig(
                **ag_config['ray_config']['rl_module']['model_config']
            )
        )
        .training(**hyperparams_config)
        .evaluation(
            **ag_config['ray_config']['evaluation'],
            custom_evaluation_function=custom_evaluation_function,
        )
        .learners(
            **ag_config['ray_config']['learners']
        )
        .reporting(
            min_time_s_per_iteration=0,
            min_train_timesteps_per_iteration=0,
            min_sample_timesteps_per_iteration=0
        )
    )

    config.validate_train_batch_size_vs_rollout_fragment_length()

    # Build the algorithm
    agent = config.build_algo()

    # Train
    for k in range(len(train_env_configs)):
        print(f'[Debug] Start training in environment {k}')

        total_timesteps = int(ag_config['train']['learning_steps'])
        n_iterations = total_timesteps // hyperparams_config['train_batch_size_per_learner']
        for i in range(n_iterations):
            start = datetime.now()
            train_res = agent.train()
            print(f'[Debug] Training iteration time seconds:{(datetime.now() - start).seconds}')
            print(f"[Debug] Iteration {i}, "
                  f"Steps: {train_res['num_env_steps_sampled_lifetime']}, "
                  f"Mean return: [Train]{train_res['env_runners']['episode_return_mean']:.2f} ")
        print(f'[Debug] End training in environment {k}')

        # Change training environment
        agent.env_runner_group.foreach_env_runner(
            func=partial(_change_env, new_task=k + 1)
            # TODO: try lambda with reset here
        )

    # Validate trained agent

    start = datetime.now()
    # val_res = agent.evaluate()  # better integration with ray but slower
    val_res = validate_serial(agent, env_id, num_episodes=5)  # custom and faster serial evaluation
    print(f'Validation time seconds:{(datetime.now() - start).seconds}, mean return: {val_res["val_episode_return_mean"]}')

    # Save checkpoint and store train() results
    with tempfile.TemporaryDirectory() as tempdir:
        agent.save_checkpoint(tempdir)
        agent.metrics.set_value('val_episode_return_mean', val_res['val_episode_return_mean'])
        train_res['val_episode_return_mean'] = val_res['val_episode_return_mean']

        tune.report(metrics=train_res, checkpoint=Checkpoint.from_directory(tempdir))

    agent.stop()



def objective_val_train(hyperparams_config):
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get training set of environments
    with open(f'{os.path.dirname(os.path.abspath(__file__))}/datasets/train/{env_id}.yaml') as f:
        train_env_configs = yaml.safe_load(f)

    # Configure the algorithm.
    config = (
        get_trainable_cls(ag_config['method']).get_default_config()
        .environment(
            env_id,
            env_config=train_env_configs[0]
        )
        .callbacks(EnvTaskCallback)
        .env_runners(
            **ag_config['ray_config']['env_runners']
        )
        .rl_module(
            model_config=DefaultModelConfig(
                **ag_config['ray_config']['rl_module']['model_config']
            )
        )
        .training(**hyperparams_config)
        .evaluation(
            **ag_config['ray_config']['evaluation'],
            custom_evaluation_function=custom_evaluation_function,
        )
        .learners(
            **ag_config['ray_config']['learners']
        )
        .reporting(
            min_time_s_per_iteration=0,
            min_train_timesteps_per_iteration=0,
            min_sample_timesteps_per_iteration=0
        )
    )

    config.validate_train_batch_size_vs_rollout_fragment_length()

    # Build the algorithm
    agent = config.build_algo()

    # Train
    all_return_mean = []
    for k in range(len(train_env_configs)):
        print(f'[Debug] Start training in environment {k}')

        agent = config.build_algo()

        total_timesteps = int(ag_config['train']['learning_steps'])
        n_iterations = total_timesteps // hyperparams_config['train_batch_size_per_learner']
        for i in range(n_iterations):
            train_res = agent.train()

        # Change training environment
        agent.env_runner_group.foreach_env_runner(
            func=partial(_change_env, new_task=k + 1)
            # TODO: try lambda with reset here
        )

        validation_train = validate_serial_train(agent, train_env_configs[k], num_episodes=5)
        print(f"[Debug] Environment {k} validation return mean: {validation_train['val_episode_return_mean']:.2f}")
        all_return_mean.append(validation_train['val_episode_return_mean'])

    # Save checkpoint and store results
    with tempfile.TemporaryDirectory() as tempdir:
        agent.save_checkpoint(tempdir)
        agent.metrics.set_value('val_episode_return_mean', np.mean(all_return_mean))
        train_res['val_episode_return_mean'] = np.mean(all_return_mean)

        tune.report(metrics=train_res, checkpoint=Checkpoint.from_directory(tempdir))

    agent.stop()


if __name__ == "__main__":

    # Input arguments: RL method and environment
    method = 'rppo'
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

    ray.init(local_mode=True)

    # Set a common working directory for all trials, note this allows all trials to access resources (e.g. image files)
    # in the main project directory, but still ensure trials logging in different directories, one for each trial.
    os.environ['RAY_CHDIR_TO_TRIAL_DIR'] = "0"

    # Get hyperparameters search space from configuration file
    hyperparams_space = ConfigurationSpace().from_serialized_dict(ag_config['tune']['search_space'])

    # Get default hyperparams configuration as a baseline from your algorithm…
    full_default = get_trainable_cls(ag_config['method']) \
        .get_default_config()
    # hp_keys = list(ag_config['tune']['search_space'].keys())
    default_hparams = {k: full_default[k] for k in hyperparams_space.at if 'model/' not in k}
    default_hparams.update({k: full_default.model[k.split('/')[1]] for k in hyperparams_space.at if 'model/' in k})

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
        objective_val_train,
        resources=tune.PlacementGroupFactory(ag_config['ray_config']['resources'])
    )

    # Initialize Ray Tuner
    tuner = Tuner(
        training_function,
        run_config=RunConfig(
            verbose=None,
            name=RUN_DIR.split('/')[-1],
            storage_path=ROOT_DIR,
        ),
        tune_config=TuneConfig(
            search_alg=bohb_search,
            scheduler=bohb_hyperband,
            **ag_config['tune']['config']
        )
    )

    # Restore Ray Tuner
    # tuner = Tuner.restore('/app/res/MNISTExib-v0/PPO/tune/run0', training_function)

    # Execute the hyperparams tuning
    results = tuner.fit()
