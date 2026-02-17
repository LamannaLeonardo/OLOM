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


def origin_state(env_digits, rot_degrees=180):
    return {
        'objects': {
            digit_id: {
                'rotation': 0,
                'flipped': 0,
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
    split = 'train'
    all_seeds = {'train': 123, 'val': 456, 'test': 789}
    seed = all_seeds[split]

    simplify = False
    env_id = "MNISTExib-v0"

    random.seed(seed)
    np.random.seed(seed)
    nenvs = 4
    env_ntasks = 5

    if simplify:
        env_id = f"simple{env_id}"

    tasks_path = f'../../datasets/{split}/{env_id}.yaml'
    if os.path.exists(tasks_path):
        os.remove(tasks_path)

    all_envs = []
    sampled_digit_conf = []
    for nobjs in [2, 4, 6, 8, 10]:

        for _ in range(nenvs):

            digit_types = list(np.random.choice(all_digits, nobjs, replace=not simplify))
            while digit_types in sampled_digit_conf:
                digit_types = list(np.random.choice(all_digits, nobjs, replace=not simplify))
            sampled_digit_conf.append(digit_types)

            count_digits = defaultdict(int)
            digits = []
            for d in digit_types:
                digits.append(f'{d}-{count_digits[d]}')
                count_digits[d] += 1

            # Define random initial state
            new_env = dict()
            digit_pos = list(range(len(digits)))
            random.shuffle(digit_pos)
            new_env['_state'] = random_state(digits)

            # Define random goal set
            goal_states = origin_state(digits)['objects']
            goal_states = {d_id: {k: v for k, v in goal_states[d_id].items() if k != 'pos'} for d_id in goal_states}
            new_env['goal_states'] = preprocess_goal_manygoals(digits, {'objects': goal_states})

            all_envs.append(new_env)

    with open(tasks_path, 'w') as f:
        yaml.dump(all_envs, f)



