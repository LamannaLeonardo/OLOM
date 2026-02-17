import itertools
import os
import random
from collections import defaultdict, Counter
from typing import List, Dict

import numpy as np
import yaml

all_digits = [f'd{i}' for i in range(10)]

def random_state(env_digits, rot_degrees=180):
    return {
        'objects': {
            digit_id: {
                'rotation': np.random.randint(int(360 / rot_degrees)) * rot_degrees,
                'flipped': np.random.randint(2),
                'pos': digit_pos[i]}
            for i, digit_id in enumerate(env_digits)},
        'agents': {
            'agent0': {  # Single-agent environment
                'pos': np.random.randint(len(env_digits))
            }
        }
    }


# Consider all possible goals that matches the given one, e.g. if there are two undistinguishable zeros that must be
# resp. rotated and flipped, then the state where they are resp. flipped and rotated is still a goal state
def preprocess_goal_manygoals(digits, goal_state: Dict) -> List:
    # goal_state is supposed to be a dict {'objects':{'d0-1':'rotation':..., 'flipped':...},
    #                                               {'d0-2':'rotation':..., 'flipped':...},
    #                                               {...}}
    digit_goals = []
    for d in digits:
        d_goals = []
        same_digits = [d2 for d2 in digits if d.split('-')[0] == d2.split('-')[0]]
        for d2 in same_digits:
            if (d, goal_state['objects'][d2]) not in d_goals:
                d_goals.append((d, goal_state['objects'][d2]))
        digit_goals.append(d_goals)

    goal_states = []
    for goal in itertools.product(*digit_goals):
        goal_states.append({'objects': {d: dict(s) for d, s in list(goal)}})

    # Filter out states that do not contain all goal object states
    goal_state_group = defaultdict(list)
    for d, v in goal_state['objects'].items():
        goal_state_group[d.split('-')[0]].append(frozenset(v.items()))

    filtered_goal_states = []
    for g in goal_states:
        g_group = defaultdict(list)
        for d, v in g['objects'].items():
            g_group[d.split('-')[0]].append(frozenset(v.items()))

        if (np.all([Counter(g_group[d]) == Counter(goal_state_group[d]) for d in goal_state_group])
                and g not in filtered_goal_states):
            filtered_goal_states.append(g)

    return filtered_goal_states


def preprocess_goal(digits, goal_state: Dict) -> List:
    # goal_state is supposed to be a dict {'objects':{'d0-1':'rotation':..., 'flipped':...},
    #                                               {'d0-2':'rotation':..., 'flipped':...},
    #                                               {...}}
    goal_states = [goal_state]

    return goal_states


if __name__ == '__main__':
    envs = []
    seed = 12345

    simplify = False

    env_id = "MNISTExib-v0"
    if simplify:
        env_id = f"simple{env_id}"

    random.seed(seed)
    np.random.seed(seed)
    nenvs = 4
    env_ntasks = 5
    train_path = f'../../../datasets/fine-tune/train/'
    eval_path = f'../../../datasets/fine-tune/test/'

    train_envs = dict()
    test_envs = dict()
    for nobjs in [2, 4, 6, 8, 10]:

        for _ in range(nenvs):

            digit_types = list(np.random.choice(all_digits, nobjs, replace=not simplify))
            count_digits = defaultdict(int)
            digits = []
            for d in digit_types:
                digits.append(f'{d}-{count_digits[d]}')
                count_digits[d] += 1

            new_env = {'train': dict(), 'test': list()}
            digit_pos = list(range(len(digits)))
            # Define the initial state of the single training episode
            random.shuffle(digit_pos)
            new_env['train']['_state'] = random_state(digits)

            # Define the goal set of both the training and evaluation/test episodes
            goal_found = False
            goal = None
            while not goal_found:
                goal = random_state(digits)['objects']
                goal = {d_id: {k: v for k, v in goal[d_id].items() if k != 'pos'} for d_id in goal}
                if np.all([g['objects'] != new_env['train']['_state']['objects']
                           for g in preprocess_goal_manygoals(digits, {'objects': goal})]):
                    goal_found = True
            goal_states = preprocess_goal_manygoals(digits, {'objects': goal})
            new_env['train']['goal_states'] = goal_states

            # Define the initial states of the evaluation/test episodes
            for _ in range(env_ntasks):
                # Randomly sample an initial state
                new_state = random_state(digits)
                # Check initial state is not in the goal set
                while np.all([g['objects'] == new_state['objects']
                              for g in preprocess_goal_manygoals(digits, {'objects': goal})]):
                    new_state = random_state(digits)
                new_env['test'].append({
                    '_state': new_state,
                    'goal_states': goal_states
                })

            train_envs[f"env-{len(train_envs)}"] = new_env['train']
            test_envs[f"env-{len(test_envs)}"] = new_env['test']

    os.makedirs(train_path, exist_ok=True)
    with open(f"{train_path}/{env_id}.yaml", 'w') as f:
        yaml.dump(train_envs, f)

    os.makedirs(eval_path, exist_ok=True)
    with open(f"{eval_path}/{env_id}.yaml", 'w') as f:
        yaml.dump(test_envs, f)
