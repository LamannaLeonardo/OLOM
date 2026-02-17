import os
# See https://discuss.ray.io/t/dreamer-v3-rllib-tensorflow-error/15127
# os.environ["TF_USE_LEGACY_KERAS"] = "1"

import random
import torch
import numpy as np
import yaml
import agents

from utils.config import load_ag_cfg, load_dom_cfg, load_run_cfg, get_ag_cfg, get_dom_cfg, get_run_cfg
import envs  # Currently used for registering custom environments in Gymnasium

if __name__ == "__main__":

    # approaches = ['rppo', 'impala', 'drqn', 'pal', 'oracle']
    train = True
    test = True
    approaches = ['pal']  # pal is OLOM in the paper
    domains = ['MNISTExib-v0', 'simpleMNISTExib-v0']
    # domains = ['MNISTExib-v0']
    nseeds = 5

    if train:
        for agent_type in approaches:
            for domain in domains:
                # Load agent configuration
                load_ag_cfg(f"configs/agents/{agent_type}.yaml")
                ag_cfg = get_ag_cfg()

                # Load domain configuration
                load_dom_cfg(f"configs/envs/{domain}.yaml")
                dom_cfg = get_dom_cfg()

                # Load run configuration
                load_run_cfg(f"configs/run.yaml")
                run_cfg = get_run_cfg()

                # For every random seed
                for n, seed in enumerate(range(nseeds)):

                    # Set random seed for reproducibility
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                    # Create agent instance from configuration
                    agent_cls_name = ag_cfg["agent"]
                    agent = getattr(agents, agent_cls_name)()

                    # Load training set of environments
                    with open(f'datasets/fine-tune/train/{dom_cfg["env_id"]}.yaml') as f:
                        train_set = yaml.unsafe_load(f)

                    # For every training environment
                    for train_env_id in sorted(train_set, key=lambda x: int(x.split('-')[1])):

                        print(f'\n# [Train] Env {train_env_id} - Seed {seed}:')

                        # Fine-tune the agent RL model
                        train_set[train_env_id]['seed'] = seed
                        agent.learn(
                            env_id=dom_cfg["env_id"],
                            env_kwargs=train_set[train_env_id],
                            input_hyperparams_path=None,
                            # input_hyperparams_path=f"res/{domain.replace('simple', '')}/{ag_cfg['method']}/tune/run0",
                            load_checkpoint=False,
                        )

    if test:
        for agent_type in approaches:
            for domain in domains:
                # Load agent configuration
                load_ag_cfg(f"configs/agents/{agent_type}.yaml")
                ag_cfg = get_ag_cfg()

                # Load domain configuration
                load_dom_cfg(f"configs/envs/{domain}.yaml")
                dom_cfg = get_dom_cfg()

                # Load run configuration
                load_run_cfg(f"configs/run.yaml")
                run_cfg = get_run_cfg()

                # For every random seed
                for n, seed in enumerate(range(nseeds)):

                    # Set random seed for reproducibility
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                    # Create agent instance from configuration
                    agent_cls_name = ag_cfg["agent"]
                    agent = getattr(agents, agent_cls_name)()

                    # Load test set of environments
                    with open(f'datasets/fine-tune/test/{dom_cfg["env_id"]}.yaml') as f:
                        test_set = yaml.unsafe_load(f)

                    # For every test environment
                    for test_env_id in sorted(test_set, key=lambda x: int(x.split('-')[1])):
                        for k, test_env in enumerate(test_set[test_env_id]):
                            # Fine-tune the agent RL model
                            print(f'\n# [Test] Env {test_env_id} - Episode {k} - Seed {seed}:')
                            test_env['seed'] = seed
                            train_dir = (f"res/{dom_cfg['env_id']}/{ag_cfg['method']}"
                                         f"/train/run{int(test_env_id.split('-')[1]) + len(test_set) * n}")
                            model_dir = [d for d in os.listdir(train_dir)
                                         if d.startswith(ag_cfg['method'])
                                         or d == 'models'
                                         or d.startswith('di-engine')][0]
                            model_path = f"{train_dir}/{model_dir}"
                            agent.solve(env_id=dom_cfg["env_id"],
                                        env_kwargs=test_env,
                                        input_model_path=model_path,
                                        seed=seed,
                                        max_steps=100)


