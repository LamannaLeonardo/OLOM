import copy
import random
from collections import OrderedDict
from typing import List

import networkx as nx
import numpy as np
import pomdp_py


class ObjectState(pomdp_py.ObjectState):
    def __init__(self, sc):
        objclass = 'object'
        super().__init__(objclass, {"sc": sc})
        self.sc = self.attributes["sc"]

    def __str__(self):
        return f"ObjectState({self.sc})"

    # @property
    # def sc(self):
    #     return self.attributes["sc"]

    def __sub__(self, other):
        raise NotImplementedError
        return 1 - float(self == other)


class RobotState(pomdp_py.ObjectState):
    def __init__(self, so):
        """Note: so is the integer id of the current propositional state of the robot"""
        objclass = 'robot'
        super().__init__(objclass, {"so": so})
        self.so = self.attributes["so"]

    def __str__(self):
        return f"RobotState({self.so})"

    def __repr__(self):
        return str(self)


class MNISTOOState(pomdp_py.OOState):
    def __init__(self, object_states):
        super().__init__(object_states)
        self.sc_states = {objid: self.object_states[objid] for objid in self.object_states
                         if isinstance(self.object_states[objid], ObjectState)}
        self.sc_states = OrderedDict(sorted(self.sc_states.items()))

        self.so_state = [self.object_states[objid] for objid in self.object_states
                         if isinstance(self.object_states[objid], RobotState)][0]

    def __str__(self):
        return f"MNISTOOState({self.object_states})"

    def __repr__(self):
        return str(self)


class Action(pomdp_py.Action):
    def __init__(self, name: str, types: List[str], objects: List[int]):
        self.name = name
        self.objects = objects
        self.types = types

    def __str__(self):
        return f"{self.name}({','.join([f'obj_{i}' for i in self.objects])})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)


class ObjectObservation(pomdp_py.Observation):

    NULL = None

    def __init__(self, kid, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kid = kid

    def __str__(self):
        return self.kid

    def __hash__(self):
        return hash(self.kid)

    def __eq__(self, other):
        return self.kid == other.kid


class MNISTOOObservation(pomdp_py.OOObservation):
    """Observation for MNISTExib that can be factored by objects;
    thus this is an OOObservation."""

    def __init__(self, ktuple):
        """
        ktuple: list of elements that can be either a cluster 'k' or NULL (not ObjectObservation!) for every object id.
        """
        self._hashcode = hash(frozenset(ktuple.items()))
        self.ktuple = ktuple

    def __hash__(self):
        return self._hashcode

    def __eq__(self, other):
        return self.ktuple == other.ktuple

    def __str__(self):
        return f"MNISTOOObservation({' '.join([f'{k}:{v.kid}' for k, v in self.ktuple.items()])})"

    def __repr__(self):
        return str(self)

    def for_obj(self, objid):
        if objid in self.ktuple:
            return ObjectObservation(self.ktuple[objid])
        else:
            return ObjectObservation(ObjectObservation.NULL)

    def factor(self, next_state, *params, **kwargs):
        """Factor this OO-observation by objects"""
        factored_obs = {objid: ObjectObservation(self.ktuple[objid])
                        for objid in next_state.sc_states}
        return factored_obs

    @classmethod
    def merge(cls, object_observations, next_state, *params, **kwargs):
        """Merge `object_observations` into a single OOObservation object;

        object_observation (dict): Maps from objid to ObjectObservation"""
        return MNISTOOObservation(object_observations)


class ObjectObservationModel(pomdp_py.ObservationModel):
    def __init__(self, c, obskc, obsc, noise):
        """
        noise is introduced in Pr(o | s, a) to avoid particle deprivation"""
        self._c = c
        self.obskc = obskc
        self.obsc = obsc
        self.noise = noise
        # Set O of (discrete) observations
        self.O = [ObjectObservation(k) for k in range(self.obskc.shape[-1])]
        # Add NULL observation
        self.O.append(ObjectObservation(ObjectObservation.NULL))

    def probability(self, obj_observation, next_obj_state, action, **kwargs):
        """
        Returns the probability Pr(o | s, a), for an object observation 'o',
        a next state 's' and an action 'a'.

        Args:
            obj_observation (ObjectObservation)
            next_obj_state (State)
            action (Action)
        """

        bo = kwargs.get("bo", None)
        assert bo is not None

        sc = next_obj_state.sc
        pr = sum([self.obskc[sc, obj_observation.kid] * self.obsc[self._c, so] * prso
                  for so, prso in enumerate(bo)])
        if obj_observation.kid != ObjectObservation.NULL:
            return pr
        else:
            return 1 - sum([self.obsc[self._c, so] * prso for so, prso in enumerate(bo)])

    def sample(self, next_state, action, **kwargs):
        """Returns an object observation 'o' sampled from Pr(o | s, a)"""

        so = next_state.so_state.so
        sc = next_state.sc_states[self._c].sc
        pr = self.obskc[sc, :] * self.obsc[self._c, so]

        # Add NULL observation probability
        pr = np.concatenate([pr, [1 - self.obsc[self._c, so]]])

        return random.choices(self.O, weights=pr)[0]


class MNISTObservationModel(pomdp_py.OOObservationModel):
    """Object-oriented transition model"""

    def __init__(self, obsk, obsc, noise=0.1):
        observation_models = {
            c: ObjectObservationModel(c, obsk[c], obsc, noise)
            for c in range(obsc.shape[0])
        }
        super().__init__(observation_models)

    def sample(self, next_state, action, **kwargs):
        factored_observations = super().sample(next_state, action)
        return MNISTOOObservation.merge(factored_observations, next_state)


class RobotTransitionModel(pomdp_py.TransitionModel):

    def __init__(self, To, ops_nullary, noise):
        self.To = To
        self.ops_nullary = ops_nullary
        self.noise = noise


    def probability(self, next_robot_state, state, action):

        if action.name == 'stop' or len(action.objects) > 0:
            if next_robot_state == state.so_state:
                return 1.
            else:
                return 0.

        so = state.so_state.so
        To = self.To[self.ops_nullary.index(action.name)]
        return To[so][next_robot_state.so]

    def sample(self, state, action):
        """Returns next_robot_state"""

        if action.name == 'stop' or len(action.objects) > 0:
            return state.so_state
        so = state.so_state.so
        To = self.To[self.ops_nullary.index(action.name)]
        so_next = np.random.choice(range(To.shape[0]), p=To[so])
        return RobotState(so_next)


class ObjectTransitionModel(pomdp_py.TransitionModel):

    def __init__(self, c, Tvc, obsc, obskc, ops_unary, noise):
        self.c = c
        self.Tvc = Tvc
        self.obsc = obsc
        self.obskc = obskc
        self.ops_unary = ops_unary
        self.noise = noise

    def probability(self, next_object_state, state, action, **kwargs):


        bo = kwargs.get("bo", None)
        assert bo is not None
        Pr_c = np.einsum('s,cs-> c', np.array(bo), self.obsc)

        # if action.operator == 'stop' or self.c not in action.objects:
        if action.name == 'stop' or Pr_c[self.c] == 0. or len(action.objects) == 0:
            if next_object_state == state:
                return 1.
            else:
                return 0.

        sc = state.sc
        sc_next = next_object_state.sc
        Tv = self.Tvc[self.ops_unary.index(action.name)]

        pr = sum([Tv[sc, sc_next] * self.obsc[self.c, so] * prso for so, prso in enumerate(bo)])
        if sc == sc_next:
            pr += (1 - sum([self.obsc[self.c, so] * prso for so, prso in enumerate(bo)]))

        return pr

    def sample(self, state, action):
        """Returns next_object_state"""
        assert len(action.objects) <= 1

        if action.name == 'stop' or self.c not in action.objects:
            return state.sc_states[self.c]

        so = state.so_state.so
        sc = state.sc_states[self.c].sc
        Tv = self.Tvc[self.ops_unary.index(action.name)]

        tv = Tv * self.obsc[self.c, so] + np.eye(Tv.shape[0]) * (1 - self.obsc[self.c, so])
        pr = tv[sc, :]
        sc_next = np.random.choice(range(Tv.shape[0]), p=pr)
        return ObjectState(sc_next)


class MNISTTransitionModel(pomdp_py.OOTransitionModel):
    """ Object-oriented transition model. """

    def __init__(self, model, noise=.0):
        transition_models = {
            c: ObjectTransitionModel(c, model.Tv[c], model.obsc, model.obsk[c], model.ops_unary, noise)
            for c in range(len(model.bc))
        }
        transition_models['robby'] = RobotTransitionModel(model.To, model.ops_zeroary, noise)

        super().__init__(transition_models)

    def sample(self, state, action, **kwargs):
        oostate = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs)
        return MNISTOOState(oostate.object_states)

    def argmax(self, state, action, normalized=False, **kwargs):
        oostate = pomdp_py.OOTransitionModel.argmax(self, state, action, **kwargs)
        return MNISTOOState(oostate.object_states)


class RewardModel(pomdp_py.RewardModel):

    def __init__(self, init_belief, goal_states, obsc, clusters, Tv, To):
        super(RewardModel, self).__init__()
        self.goal_states = goal_states
        self.obsc = obsc
        self.clusters = clusters
        self.Tv = Tv
        self.To = To
        self.dist_so = [[self.dijkstra(self.To, so=i, sop=j) if i != j else 0.
                        for j in range(self.To[0].shape[0])]
                        for i in range(self.To[0].shape[0])]
        self.dist_sc = [[[self.dijkstra(self.Tv[c], so=i, sop=j) if i != j else 0.
                        for j in range(self.Tv[c][0].shape[0])]
                        for i in range(self.Tv[c][0].shape[0])]
                        for c in range(len(self.Tv))]

        # Select a subset of goal state that minimizes the distance from the initial agent state.
        # Indeed, a large set of goal state can slow down the reward function computation significantly (since it
        # minimizes the distance from the current state to all goal states)
        s_init = init_belief.mpe()  # maximum a posteriori estimation
        dist_sp = [self.dist(s_init, g) for g in self.goal_states]
        goal_states_subset = []
        for _ in range(1):  # consider a subset of 1 goal state
            if len(dist_sp) == 0:
                break
            closest_goal_idx = np.argmin(dist_sp)
            goal_states_subset.append(copy.deepcopy(self.goal_states[closest_goal_idx]))
            del self.goal_states[closest_goal_idx]
            del dist_sp[closest_goal_idx]

        self.goal_states = goal_states_subset

    def dist(self, s: MNISTOOState, g: List[int]):
        # Get distance between object states
        dist_sc = []
        for c, (sc, scgoal) in enumerate(zip(s.sc_states.values(), g)):
            if sc != scgoal:
                dist_sc.append(self.dist_sc[c][sc.sc][scgoal.sc])
            else:
                dist_sc.append(0)

        # Get distance between agent position and every object to be manipulated
        so = s.so_state.so
        dist_so = [self.dist_so[so][i]
                   for i in range(len(dist_sc))
                   if dist_sc[i] > 0 and dist_sc[i] != np.inf]

        # Return overall distance
        if sum(dist_so) == 0:
            return np.mean(dist_sc)

        dist_s = sum(dist_so) + sum(dist_sc)**2

        return dist_s

    def distNEWTRIAL(self, s: MNISTOOState, g: List[int]):
        # Get distance between object states
        so = s.so_state.so
        dist_sc = []
        for c, (sc, scgoal) in enumerate(zip(s.sc_states.values(), g)):
            if sc != scgoal:
                dist_sc.append(self.dist_sc[c][sc.sc][scgoal.sc])
            else:
                dist_sc.append(0)

        # Get distance between agent position and every object to be manipulated
        dist_so = [self.dist_so[so][i]
                   for i in range(len(dist_sc))
                   if dist_sc[i] > 0 and dist_sc[i] != np.inf]

        # Return overall distance
        if sum(dist_so) == 0:
            return np.mean(dist_sc)

        dist_s = sum(dist_so) + sum(dist_sc)**2

        return dist_s

    def _reward_func(self, s, a, sp):

        # Get minimum distance between states in the goal set
        dist_sp = min([self.dist(sp, g) for g in self.goal_states])

        ndigits = len(s.sc_states)
        if a.name == 'stop':
            if dist_sp == 0:
                return 100 * ndigits
            else:
                return -100 * ndigits

        reward = - dist_sp

        return reward

    def _reward_funcNEWTRIAL(self, s, a, sp):

        # Get minimum distance between states in the goal set
        dist_sp = min([self.dist(sp, g) for g in self.goal_states])

        ndigits = len(s.sc_states)

        if a.name == 'stop':
            if dist_sp == 0:
                return 100 * ndigits
            else:
                return -100 * ndigits

        reward = - dist_sp

        return reward


    def dijkstra(self, matrices, so, sop):

        # Create a graph
        G = nx.DiGraph()  # Use nx.Graph() for an undirected graph

        # Add weighted edges (node1, node2, weight)
        edges = [(i, j, T[i, j])
                 for T in matrices
                 for i in range(T.shape[0])
                 for j in range(T.shape[0])
                 if j == np.argmax(T[i, :])
                 ]
        # Add edges to the graph with transformed weights
        for u, v, prob in edges:
            G.add_edge(u, v, weight=1)

        # Compute shortest path
        start_node = so
        end_node = sop
        try:
            weighted_path_length = nx.dijkstra_path_length(G, source=start_node, target=end_node, weight='weight')
        except:
            return np.inf

        return weighted_path_length

    def sample(self, state, action, next_state):
        return self._reward_func(state, action, next_state)

    def argmax(self, state, action, next_state):
        return self._reward_func(state, action, next_state)


class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a
       small, finite action space"""

    def __init__(self, obsc, operators):
        super(PolicyModel, self).__init__()
        self.obsc = obsc
        self.operators = operators

    def sample(self, state):
        return random.sample(self.get_all_actions(state), 1)[0]

    def rollout(self, state, *args):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state, history=None):
        so = state.so_state.so
        objs = np.nonzero(self.obsc[:, so])[0]
        all_actions = [Action(o.name, o.types, []) for o in self.operators if o.arity == 0]

        for op in [o.name for o in self.operators if o.arity == 1]:
            for obj in objs:
                all_actions.append(Action(op, ['object'], [obj]))

        return all_actions


class MNISTOOBelief(pomdp_py.OOBelief):
    """This is needed to make sure the belief is sampling the right
    type of State for this problem."""

    def __init__(self, object_beliefs):
        """
        object_beliefs (objid -> GenerativeDistribution), note this includes robot (propositional) belief state
        """
        super().__init__(object_beliefs)

    def mpe(self, **kwargs):
        return MNISTOOState(pomdp_py.OOBelief.mpe(self, **kwargs).object_states)

    def random(self, **kwargs):
        return MNISTOOState(pomdp_py.OOBelief.random(self, **kwargs).object_states)
