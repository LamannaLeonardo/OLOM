import copy
import itertools
import random
from typing import List
from gymnasium import spaces
import numpy as np

import networkx as nx
from networkx.algorithms.approximation import traveling_salesman_problem

from PAL.Modeling.Operator import Operator
from envs.TaskableEnv import TaskableEnv, StateType, ObsType


class MNISTExib(TaskableEnv):
    """
    The MNIST-Exib environment, where an agent navigates an exhibition of MNIST pictures.
    At each location, the agent perceives a discrete observation corresponding to the 1-hot encoding of the
    cluster id associated with the RGB image of the MNIST digit in front of the agent.
    The agent can navigate to the right/left location, and rotate/flip the picture in
    front of it. The goal is to rotate/flip all digits in some configuration, and finally call a "stop" action
    to end the episode.
    """

    metadata = {"render_modes": ["human"]}

    ROT_DEGREES: int = 180
    ACT_NOISE: float = .15  # actuation noise
    CLUSTER_NOISE: float = .2  # actuation noise

    operators: List[Operator] = [
        Operator(name='stop', types=[]),
        Operator(name='right', types=[]),
        Operator(name='left', types=[]),
        Operator(name='rotate', types=['object']),
        Operator(name='flip', types=['object']),
    ]

    n_digits = 10
    n_states_per_digit = 4  # (not) flipped and (not) rotated

    def __init__(self, **kwargs):

        # Initialize environment state and goal set
        super(MNISTExib, self).__init__(**kwargs)

        # Define the observation space: RGB 28x28 images
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.n_digits * self.n_states_per_digit,),  # img_shape
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(n=len(self.operators))

        # Store images of rotated digits to speed up :meth: step
        assert self.ROT_DEGREES == 180, 'Changing rotation degrees requires revising hard-coded digit states'
        one_hot_matrix = np.eye(self.n_digits * self.n_states_per_digit, dtype=np.float32)

        self.digit_clusters = {
            f'd{i}': {
                0: {  # not flipped
                    0: one_hot_matrix[0 + self.n_states_per_digit * i],  # not rotated
                    180: one_hot_matrix[1 + self.n_states_per_digit * i]   # rotated
                },
                1: {  # flipped
                    0: one_hot_matrix[2 + self.n_states_per_digit * i],  # not rotated
                    180: one_hot_matrix[3 + self.n_states_per_digit * i]   # rotated
                }
            }
            for i in range(self.n_digits)
        }

        # Add clustering noise
        self.all_clusters = [one_hot_matrix[self.n_states_per_digit * i + j]
                             for j in range(4)
                             for i in {int(d.split('-')[0].replace('d', '')) for d in self._state['objects']}]

        # Inject noise
        random.seed(123)  # ensure reproducible results across runs
        for d_key in self._state['objects']:

            d_key = d_key.split('-')[0]
            digit_clusters = [one_hot_matrix[self.n_states_per_digit * int(d_key.replace('d', '')) + j]
                              for j in range(4)]
            flip_dict = self.digit_clusters[d_key]
            for flip, rot_dict in flip_dict.items():
                for rot, vec in rot_dict.items():
                    if random.random() < self.CLUSTER_NOISE:
                        wrong_vec = random.choice(self.all_clusters)  # allow mixing clusters of different digits
                        # wrong_vec = random.choice(digit_clusters)  # allow mixing clusters only of the same digit
                        print(f'wrong cluster assigned to {d_key} f{flip} r{rot}')
                        self.digit_clusters[d_key][flip][rot] = wrong_vec
        random.seed(self.seed)

        self.digits = {d: self.digit_clusters[d.split('-')[0]] for d in self._state['objects']}

        # Compute (possibly) superset of goal states
        self.goal_states_superset = self.goal_superset(self.goal_states)
        # self.goal_states = [self.goal_states[0]]

    def _randomize_state(self) -> None:
        assert self._state is not None, (f"{self.__module__} environment must be initialized with some state "
                                         f"in order to randomize its state")
        # Randomize object states
        for d_id in self._state['objects']:
            self._state['objects'][d_id]['rotation'] = (np.random.randint(int(360 / self.ROT_DEGREES)) * self.ROT_DEGREES)
            self._state['objects'][d_id]['flipped'] = np.random.randint(2)

        # Randomize agent state, i.e. its position
        self._state['agents']['agent0']['pos'] = np.random.randint(len(self._state['objects']))


    def _get_obs(self) -> ObsType:
        """
        Render the current state of the environment.
        """

        # Get the digit in the current agent view
        digit_pos = self._state['agents']['agent0']['pos']
        digit_id = next(d_id for d_id in self._state['objects']
                        if self._state['objects'][d_id]['pos'] == digit_pos)

        flipped = self._state['objects'][digit_id]['flipped']
        rotation = self._state['objects'][digit_id]['rotation']

        cluster = self.digits[digit_id][flipped][rotation]

        return cluster

    def get_state_rgbs(self, state: StateType) -> List[np.ndarray]:
        assert state is not None, 'Cannot get the RGB images of a None state'
        state_rgbs = list()
        for digit_id in state['objects']:
            rgb = self.digits[digit_id.split('-')[0]]

            digit_rgb = copy.deepcopy(rgb)

            # Flip the digit
            if state['objects'][digit_id]['flipped'] == 1:
                digit_rgb = np.flipud(digit_rgb)

            # Rotate the digit
            for k in range(state['objects'][digit_id]['rotation'] // 90):
                digit_rgb = np.rot90(digit_rgb)

            # Store the digit RGB image
            state_rgbs.append(digit_rgb)

        return state_rgbs

    def get_state_rgbs_samples(self, state):
        assert state is not None, 'Cannot get the RGB images of a None state'
        samples = []

        for digit_id in state['objects']:

            # Flip the digit
            flipped = state['objects'][digit_id]['flipped']
            rotation = state['objects'][digit_id]['rotation']
            digit_rgb = self.digits[digit_id][flipped][rotation]

            sample = {'rgb': digit_rgb,
                      # for simulation and evaluation
                      'eval': {'digit': digit_id,
                               'state': state}
                      }
            samples.append(sample)

        return samples

    def get_state_obs(self, state):

        state_obs = [None for _ in range(len(self.digits))]
        for digit_id in state['objects']:
            flipped = state['objects'][digit_id]['flipped']
            rotation = state['objects'][digit_id]['rotation']
            state_obs[int(self._state['objects'][digit_id]['pos'])] = self.digits[digit_id][flipped][rotation]

        return state_obs

    def get_goal_obs(self):

        all_goal_obs = {frozenset(tuple(e) for e in self.get_state_obs(g)) for g in self.goal_states}

        assert len(all_goal_obs) == 1, ('The set of goal states contains states that can produce different '
                                        'observations. This is not yet allowed: disjunctive goals are not implemented')

        return self.get_state_obs(self.goal_states[0])

    def dist_heuristic(self, s, g):

        # Get distance between object states, note this only considers
        # object states mentioned in the goal set
        dist_sc = {
            c: sum([s['objects'][c][sc] != g['objects'][c][sc]
                    for sc in s['objects'][c] if sc in g['objects'][c]])
            for c in s['objects']
        }

        # Get distance between agent position and every object to be manipulated
        so = s['agents']['agent0']['pos']
        dist_so = {
            c: abs(s['objects'][c]['pos'] - so) * int(dist_sc[c] > 0)
            for c in s['objects']
        }

        if sum(dist_so.values()) == 0:
            return sum(dist_sc.values())
        dist_s = sum(dist_so.values()) + sum(dist_sc.values())**2

        return dist_s

    # Consider all possible goals that matches the given one, e.g. if there are two undistinguishable zeros that must be
    # resp. rotated and flipped, then the state where they are resp. flipped and rotated is still a goal state
    def goal_superset(self, goal_states) -> List:
        # goal_state is supposed to be a dict {'objects':{'d0-1':'rotation':..., 'flipped':...},
        #                                               {'d0-2':'rotation':..., 'flipped':...},
        #                                               {...}}

        goal_states_superset = []
        for goal_state in goal_states:
            new_goal_states_superset = []
            digit_goals = []
            for d in self.digits:
                d_goals = []
                same_digits = [d2 for d2 in self.digits if d.split('-')[0] == d2.split('-')[0]]
                for d2 in same_digits:
                    if (d, goal_state['objects'][d2]) not in d_goals:
                        d_goals.append((d, goal_state['objects'][d2]))
                digit_goals.append(d_goals)

            for goal in itertools.product(*digit_goals):
                new_goal_states_superset.append({'objects': {d: dict(s) for d, s in list(goal)}})

            # Filter out states that do not contain all goal object states
            new_goal_states_superset = [g for g in new_goal_states_superset
                                        if ({frozenset(obj_s.items()) for obj_s in g['objects'].values()}
                                        == {frozenset(obj_s.items()) for obj_s in goal_state['objects'].values()})
                                        and g not in goal_states_superset]
            goal_states_superset.extend(new_goal_states_superset)

        return goal_states_superset

    def _reward_fn(self, s, op: int, sp) -> np.float32:

        # Get manipulation distance between previous agent state and states in the goal set
        dist_s = min([self.dist_manip(s, g) for g in self.goal_states])

        # Get manipulation distance between current agent state and states in the goal set
        dist_sp = min([self.dist_manip(sp, g) for g in self.goal_states])

        reward = -0.1  # action execution cost

        # If the agent is stopping in a goal state
        if self.operators[op].name == 'stop':
            if dist_sp == 0:
                return reward + 10

        # If the agent gets closer to a goal state
        if dist_s - dist_sp > 0:
            return reward + 2

        # If the agent gets further from a goal state
        elif dist_s - dist_sp < 0:
            return reward - 2

        # Cost for executing an action
        else:
            return reward


    def step(self, op_id: int):
        """
        Executes a step in the environment based on the action.
        """
        self.current_step += 1

        _state_prev = copy.deepcopy(self._state)

        assert op_id >= 0

        if not random.random() < self.ACT_NOISE:
            # Noop or stop action
            if op_id is None or self.operators[op_id].name in ['stop', 'noop']:
                pass
            # Left
            elif self.operators[op_id].name == 'left':
                self._state['agents']['agent0']['pos'] -= 1
                self._state['agents']['agent0']['pos'] = max(0, self._state['agents']['agent0']['pos'])
            # Right
            elif self.operators[op_id].name == 'right':
                self._state['agents']['agent0']['pos'] += 1
                self._state['agents']['agent0']['pos'] = min(len(self._state['objects']) - 1, self._state['agents']['agent0']['pos'])
            # Rotate
            elif self.operators[op_id].name == 'rotate':
                digit_id = next(d_id for d_id in self._state['objects']
                                if self._state['objects'][d_id]['pos'] == self._state['agents']['agent0']['pos'])
                self._state['objects'][digit_id]['rotation'] += self.ROT_DEGREES
                self._state['objects'][digit_id]['rotation'] %= 360
            # Flip
            elif self.operators[op_id].name == 'flip':
                digit_id = next(d_id for d_id in self._state['objects']
                                if self._state['objects'][d_id]['pos'] == self._state['agents']['agent0']['pos'])
                self._state['objects'][digit_id]['flipped'] += 1
                self._state['objects'][digit_id]['flipped'] %= 2
            else:
                raise NotImplementedError
        else:
            # print('[Debug] Execution failure occurring')
            pass

        # Compute the reward as the negative difference between the normalized RGB images in the
        # current environment and goal state
        reward = self._reward_fn(_state_prev, op_id, self._state)
        done = (min([self.dist_heuristic(self._state, g) for g in self.goal_states]) == 0) and self.operators[op_id].name == 'stop'
        truncated = (self.current_step >= self.max_steps) or ((self.operators[op_id].name == 'stop') and (not done))

        # Additional info
        info = self._get_info()

        return self._get_obs(), reward, done, truncated, info  # Next state, reward, done, truncated, info


    def dist_manip(self, s, g):
        """
        Manipulation distance between two states.
        :param s: current state
        :param g: goal state
        :return: manipulation cost
        """
        manipulation_cost = 0.
        for d in s['objects']:
            manipulation_cost += int(s['objects'][d]['flipped'] != g['objects'][d]['flipped'])
            manipulation_cost += abs((s['objects'][d]['rotation'] - g['objects'][d]['rotation']) // self.ROT_DEGREES)

        return manipulation_cost


    def dist_min(self, s, g):
        """
        Minimum distance between two states, note this can be computationally expensive 
        since computes the optimal plan for going from state 's' to state 'g'.
        :param s: current state
        :param g: goal state
        :return: optimal plan cost
        """

        digits_pos = [s['agents']['agent0']['pos']]
        manipulation_cost = 0.
        for d in s['objects']:
            manipulation_cost += int(s['objects'][d]['flipped'] != g['objects'][d]['flipped'])
            manipulation_cost += abs((s['objects'][d]['rotation'] - g['objects'][d]['rotation']) // self.ROT_DEGREES)

            if manipulation_cost > 0:
                d_pos = s['objects'][d]['pos']
                digits_pos.append(d_pos)

        # Create a graph
        G = nx.complete_graph(len(digits_pos))
        weights = dict()
        for i in range(len(digits_pos)):
            for j in range(len(digits_pos)):
                distance = abs(digits_pos[i] - digits_pos[j])
                weights[(i, j)] = distance

        # Add weights to edges
        nx.set_edge_attributes(G, weights, "weight")

        # Solve TSP
        if len(G.nodes) == 1:
            return manipulation_cost
        tour = traveling_salesman_problem(G, cycle=False, weight="weight")

        # Calculate the total expected cost of the optimal plan
        navigation_cost = sum(G[tour[i]][tour[i + 1]]['weight'] for i in range(len(tour) - 1))
        total_cost = navigation_cost + manipulation_cost
        return total_cost

    def distance_to_success(self) -> float:
        """
        Return the optimal plan cost (i.e. length)
        """
        return min([self.dist_min(self._state, g) for g in self.goal_states])
