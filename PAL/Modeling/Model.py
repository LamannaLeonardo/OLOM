import copy
import pickle
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Dict, Any

import numpy as np
import pomdp_py

from sklearn.preprocessing import normalize

from PAL.Modeling.Object import Object
from PAL.Modeling.Operator import Operator

from utils.oopomdp import ObjectState, RobotState, MNISTOOBelief, PolicyModel, MNISTTransitionModel, \
    MNISTObservationModel, RewardModel


@dataclass
class Model:

    operators: List[Operator]

    # Set of objects
    objects: list = field(default_factory=list)

    # Initialize the empty set of clusters
    clusters: dict = field(default_factory=dict)

    # Cluster observation function, i.e., matrix Sv x K with the number Sv of unary
    # predicates associated with the set of operators OP, C of objects and K of object clusters
    obsk: np.ndarray = field(default_factory=lambda: np.empty((0, 1), dtype=float))

    # Matrix Sv x C with the number Sv of unary predicates associated with the set of
    # operators OP and C of objects indicating the probability of an object in C being in an (object)
    # belief state in Sv
    bc: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # Vector with a number |So| of nullary predicates indicating the probability of being in every agent
    # (propositional) belief state in So
    bo: np.ndarray = field(default_factory=lambda: np.zeros((0)))

    # Object observation function, i.e. a matrix C x Sag indicating the probability of observing an object c in C
    # while being in an agent (propositional) state in So
    obsc: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # List of matrices (one for every op in OP) So x So with the number So of nullary predicates
    To: list = field(init=False)  # set in __post_init__
    WTo: list = field(init=False)  # set in __post_init__

    # List of matrices (one for every op in OP) Sv^op x Sv^op with the set Sv^op of unary predicates associated
    # with operator op
    Tv: list = field(init=False)  # set in __post_init__
    WTv: list = field(init=False)  # set in __post_init__

    # Probability of observing an object c in the current belief state, i.e. Pr(c)
    # Pr_c: any = None
    Pr_c: np.ndarray = field(default_factory=lambda: np.empty(0))

    # Probability of observing an object c given a belief state sv
    WPr_c_sv: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    Pr_c_sv: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    # Some noise is introduced in the transition function because when the observation and transition
    # functions fully disagree (e.g. due to an action failure), then the belief state updated by multiplying
    # them cannot be zero everywhere.
    noise: float = np.random.uniform(0, 0)
    transition_lr: float = 1.

    # Discretization function mapping every observation to a discrete one (i.e. a cluster)
    discrete_obs: Dict[Any, int] = field(default_factory=dict)

    def __post_init__(self):
        # List of matrices (one for every op in OP) So x So with the number So of nullary predicates
        self.To = [np.zeros((1, 1)) for _ in self.ops_zeroary]
        self.WTo = [np.zeros((1, 1)) for _ in self.ops_zeroary]

        # List of matrices (one for every op in OP) Sv^op x Sv^op with the set Sv^op of unary predicates associated
        # with operator op
        self.Tv = [np.zeros((0, 0)) for _ in self.ops_unary]
        self.WTv = [np.zeros((0, 0)) for _ in self.ops_unary]

    @cached_property
    def ops_zeroary(self):
        return [o.name for o in self.operators if o.arity == 0]

    @cached_property
    def ops_unary(self):
        return [o.name for o in self.operators if o.arity == 1]

    def add_agent_state(self) -> None:
        # Initialize probability distribution for a new agent (propositional) state so' such that Pr(so') = 1
        self.bo = np.concatenate([np.zeros(self.bo.shape), np.ones(1)])

    def add_lifted_state(self, k: int) -> None:
        # Extend the cluster observation function (i.e. a Sv x K matrix) with the new lifted
        # state sv' probability distribution Pr(k | sv') = 1
        new_state_obs = np.zeros((1, self.obsk.shape[1]))
        new_state_obs[0, k] = 1.
        self.obsk = np.concatenate([self.obsk, new_state_obs], axis=0)

        # Extend the belief state Sv x C
        self.bc = np.vstack([self.bc, np.zeros(self.Pr_c.shape[0])])

        # Extend probability of C given Sv, i.e. C x Sv
        self.Pr_c_sv = np.vstack([self.Pr_c_sv, np.zeros(self.Pr_c.shape[0])])
        self.WPr_c_sv = np.vstack([self.WPr_c_sv, np.zeros(self.Pr_c.shape[0])])

        # Expand the transition function with the newly created 'lifted' state sv
        for o in range(len(self.Tv)):
            self.Tv[o] = np.pad(self.Tv[o], (0, 1))
            self.WTv[o] = np.pad(self.WTv[o], (0, 1))

            # Randomly initialize transition function
            for s in range(self.Tv[o].shape[0]):
                if (np.all([self.Tv[o][s, sp] == self.Tv[o][s, :-1].mean()
                            for sp in range(self.Tv[o].shape[1] - 1)])):
                    self.Tv[o][s, :] = np.ones((1, self.Tv[o].shape[1])) * 1 / self.Tv[o].shape[1]

    def add_object(self, k: int) -> None:

        # Initialize cluster probability distributions for the new lifted state, with a vector Sv x K
        if self.obsk[:, k].sum() == 0:
            # Create a new state sv
            sv = self.obsk.shape[0]
            self.add_lifted_state(k)
        else:
            sv = self.obsk[:, k].argmax()  # Use an existing state sv that matches the cluster
            self.obsk[sv, k] += 1.
            # Update the cluster observation function (i.e. a Sv x K matrix) by increasing Pr(k | sv)
            self.obsk = normalize(self.obsk, axis=1, norm='l1')

        # Initialize the new object belief state Sv x 1(C)
        new_obj_states = np.zeros((self.obsk.shape[0], 1))
        new_obj_states[sv, 0] = 1.  # b_c(s_v) = 1 for the new object c and s_v in S_v
        self.bc = np.concatenate((self.bc, new_obj_states), axis=1)

        # Initialize the object observation function obs_c with the probability of observing the new
        # object c while being in the current belief (propositional) state b_o
        self.obsc = np.pad(self.obsc, (0, 1))
        self.obsc[-1, :] = self.bo

        # Extend probability of c given sv
        self.Pr_c_sv = np.hstack((self.Pr_c_sv, np.zeros(new_obj_states.shape)))
        self.WPr_c_sv = np.hstack((self.WPr_c_sv, np.zeros(new_obj_states.shape)))

        assert np.all(self.obsc.sum(axis=-1) > 1 - 1e-5)
        assert np.all(self.obsc.sum(axis=-1) < 1 + 1e-5)

    def to_new_clusters(self, M):
        # Map the observation function Sc x K into the space Sc x K', where K is the previous
        # set of clusters and K' is the current (new) set of clusters.
        self.obsk = np.matmul(self.obsk, M)
        self.obsk = normalize(self.obsk, axis=1, norm='l1')

    def init_mdp(self, trace):
        # Initialize the observation function from a trace with a single observation, a single cluster, a single object,
        # and no actions.
        k = trace.observations[0]

        # Create the initial agent (propositional) state s_o, and set b_o(s_o) = 1
        self.add_agent_state()

        # Create a new object c in a new state sv and cluster k, initializes obs_c(c, so) = 1 for the only so and
        # obs_k(sv, k) = 1 for the only sv and k
        self.add_object(k)

        # Update probability of C given S
        self.Pr_c_sv = self.bc
        self.WPr_c_sv = self.bc

    def init(self, k: int):
        """
        Initialize the OPO-RMDP from a single discrete observation. This method creates:
        - the first propositional state of the agent
        - the first constant identifying the observed object
        - the first (observed) object state
        :param obs: the current discrete observation
        """

        # Create the initial agent (propositional) state s_o, and set b_o(s_o) = 1
        self.add_agent_state()

        # Create a new object c in a new state sv and cluster k, initializes obs_c(c, so) = 1 for the only so and
        # obs_k(sv, k) = 1 for the only sv and k
        self.add_object(k)

        # Update probability of C given S
        self.Pr_c_sv = self.bc
        self.WPr_c_sv = self.bc

    def set_current_state(self, k_id: int) -> None:
        nobjs = self.obsc.shape[0]
        nstates = self.obsk.shape[0]
        Pr_c = np.einsum('s, sc -> c', self.obsk[:, k_id], self.Pr_c_sv)
        self.bc = np.zeros((nstates, nobjs))
        for c in range(nobjs):
            # If object is not visible, then uniformly initialize its belief state,
            # otherwise take into account the object state predicted by the object observation function
            # and the probability of observing the object given by considering all object observation functions
            bc_random = np.array([1. if self.Pr_c_sv[sv, c] > 0 else 0. for sv in range(nstates)])
            bc_random /= bc_random.sum()
            bc_pred = self.obsk[:, k_id]
            assert 1 - 1e-5 < bc_random.sum() < 1. + 1e-5, 'Normalize initialization of belief object states'
            assert 1 - 1e-5 < bc_pred.sum() < 1. + 1e-5, 'Normalize initialization of belief object states'
            self.bc[:, c] = bc_random * (1 - Pr_c[c]) + bc_pred * Pr_c[c]
            assert 1 - 1e-5 < self.bc[:, c].sum() < 1. + 1e-5, 'Normalize initialization of belief object states'
            # Avoid numerical errors by rescaling of an epsilon quantity the probability
            # for the last object state
            max_sc = self.bc[:, c].argmax()
            self.bc[max_sc, c] = 1.0 - (self.bc[:, c].sum() - self.bc[max_sc, c])

        self.bo = np.einsum('c, cs -> s', Pr_c, self.obsc)

    def update(self, trace, update_transf=True):

        # Process every pair <k, a, k'>, where k, k' in K and a is an (either 0-ary or 1-ary) ground action
        for i in range(len(trace.observations) - 1):
            action = trace.actions[i]
            k_prev = trace.observations[i]
            k_next = trace.observations[i+1]

            # Probability of observing an object c in the current belief state. This depends only on the
            # propositional state of the agent
            self.Pr_c = np.einsum('s, cs -> c', self.bo, self.obsc)
            assert 1 + 1e-1 > self.Pr_c.sum() > 1 - 1e-1

            # Probability of observing a cluster k when observing an object c in its current
            # object belief state, i.e. Pr(k | sv, c) * Pr(sv, c).
            # Note this does not consider the probability of observing c in the current propositional state
            Pr_k_bc = np.einsum('s, sc -> c', self.obsk[:, k_next], self.bc)

            # DEBUG
            # print(f"[Debug] {i}: {action}")

            # If the executed action is a nullary operator
            if action.arity == 0:
                op = self.ops_zeroary.index(action.name)
                # When the next cluster k' is a new one, since the executed action involves no object, then create
                # a new object c and initialize the observation function such that Pr(k'|c) = 1.
                # Moreover, since the agent is looking at a new object, extend the belief propositional state with a
                # new agent state sag such that Pr(c|sag) = 1.
                if Pr_k_bc.sum() < 1e-15:

                    # Store the previous belief (propositional) state
                    bo_prev = copy.deepcopy(self.bo)

                    # Create a new agent (propositional) state so and set b(so) = 1
                    self.add_agent_state()

                    # Create a new object c
                    self.add_object(k_next)

                    # Extend the transition function of every nullary operator with a new dimension to include the
                    # new propositional state so
                    self.To = [np.pad(t, (0, 1)) for t in self.To]
                    self.WTo = [np.pad(t, (0, 1)) for t in self.WTo]

                    # Randomly initialize the transition function
                    for o in range(len(self.To)):
                        for s in range(self.To[o].shape[0]):
                            if (np.all([self.To[o][s, sp] == self.To[o][s, :-1].mean()
                                        for sp in range(self.To[o].shape[1] - 1)])):
                                self.To[o][s, :] = np.ones((1, self.To[o].shape[1])) * 1 / self.To[o].shape[1]

                    # Update the transition function of the executed nullary operator
                    self.To[op][:bo_prev.shape[0], -1] = bo_prev
                    self.WTo[op][:bo_prev.shape[0], -1] += bo_prev

                    # Normalize the transition function
                    self.To[op] = normalize(self.To[op], axis=1, norm='l1')

                # If the cluster k' is not a new one, since the executed action involves no object then the agent
                # must be looking at an already known object c
                else:

                    # Update the transition function according to the observation function
                    if update_transf:

                        additive_term = np.einsum('s, c, cp -> sp', self.bo, Pr_k_bc, self.obsc)
                        self.WTo[op] += additive_term

                        # Weight transition function update
                        self.To[op] += np.einsum('sp, sp -> sp',
                                                 additive_term,
                                                 normalize(self.WTo[op], axis=1, norm='l1'))

                    # Normalize the transition function
                    self.To[op] = normalize(self.To[op], axis=1, norm='l1')

                    # Update the agent belief state according to the revised transition function
                    self.bo = np.einsum('sp, s, pc, c -> p', self.To[op], self.bo, self.obsc, Pr_k_bc)
                    self.bo /= self.bo.sum()

                    # Update and normalize the object observation function obs_c
                    self.Pr_c = np.einsum('s, cs -> c', self.bo, self.obsc)
                    # self.obsc += np.einsum('s, c, c -> cs', self.bo, Pr_k_bc, self.Pr_c) ?
                    # self.obsc = normalize(self.obsc, axis=1, norm='l1')

                    # Update belief state according to observation function. Note the object transition function
                    # cannot be considered here since the agent executed a zeroary action
                    bc_pred = np.einsum('sc, s -> sc', self.bc, self.obsk[:, k_next])

                    bc_pred = normalize(bc_pred, axis=0, norm='l1')
                    bc_prev = copy.deepcopy(self.bc)

                    if np.max(bc_pred) == 1.:
                        self.bc = self.bc * (1 - self.Pr_c) + bc_pred * self.Pr_c
                        self.bc = normalize(self.bc, axis=0, norm='l1')
                        if not np.all((1 - 1e-5 < self.bc.sum(axis=0)) & (self.bc.sum(axis=0) < 1 + 1e-5)):
                            breakpoint()
                        assert np.all((1 - 1e-5 < self.bc.sum(axis=0)) & (self.bc.sum(axis=0) < 1 + 1e-5))
                    else:
                        pass
                        print("[Debug] #1: belief state unchanged due to disagreement with transition function")

            # If the executed action is a unary operator
            else:
                op = self.ops_unary.index(action.name)

                # Store the previous belief (propositional) state
                bc_prev = copy.deepcopy(self.bc)

                # If the cluster is a new one, since the executed operator involves an object c, then create
                # a new object state sc;
                # Then, initialize the observation function obs(k'|sc) and extend the (object) belief
                # state with sc.
                if self.obsk[:, k_next].sum() == 0:

                    # Create a new object state
                    self.add_lifted_state(k_next)

                    # Since a new state sv' has been added to Sv, then update the transition function according
                    # to the previous belief state, i.e. T(sv, op, sv') = sum_c { bc(sv) * Pr(c) }
                    self.Tv[op][:bc_prev.shape[0], -1] = (bc_prev * self.Pr_c).sum(axis=-1)
                    self.WTv[op][:bc_prev.shape[0], -1] += (bc_prev * self.Pr_c).sum(axis=-1)

                    # Normalize transition function
                    self.Tv[op] = normalize(self.Tv[op], axis=1, norm='l1')

                    # Update objects belief state
                    bcnew = np.zeros(self.bc.shape)
                    bcnew[-1, :] = self.Pr_c
                    self.bc = self.bc * (1 - self.Pr_c) + bcnew

                # If the cluster is a not a new one, since the executed operator is unary, then update
                # its transition function
                else:

                    # Update transition function according to belief and observation function
                    if update_transf:

                        additive_term = np.einsum('sc, c, p -> sp', self.bc, self.Pr_c, self.obsk[:, k_next])
                        self.WTv[op] += additive_term

                        # Weight transition function update
                        self.Tv[op] += np.einsum('sp, sp -> sp',
                                                 additive_term,
                                                 normalize(self.WTv[op], axis=1, norm='l1'))

                        # Normalize transition function
                        self.Tv[op] = normalize(self.Tv[op], axis=1, norm='l1')


                    # Update belief state according to revised transition function and observation function.
                    bc_pred = np.einsum('sc, sp, p -> pc', self.bc, self.Tv[op], self.obsk[:, k_next])
                    bc_pred = normalize(bc_pred, axis=0, norm='l1')

                    # Check if predicted belief state according to observation and transition function is certain enough
                    # otherwise there can be numerical problems when the bc is very low due to disagreement between
                    # transition and observation function. In such case do not modify the belief state
                    if np.max(bc_pred) == 1.:
                        self.bc = self.bc * (1 - self.Pr_c) + bc_pred * self.Pr_c
                        self.bc = normalize(self.bc, axis=0, norm='l1')
                        if not np.all((1 - 1e-5 < self.bc.sum(axis=0)) & (self.bc.sum(axis=0) < 1 + 1e-5)):
                            breakpoint()
                        assert np.all((1 - 1e-5 < self.bc.sum(axis=0)) & (self.bc.sum(axis=0) < 1 + 1e-5))
                    else:
                        pass
                        print("[Debug] #2: belief state unchanged due to disagreement with transition function")

                # Update observation function according to revised belief
                self.obsk[:, k_next] += np.einsum('sc, c, s -> s', self.bc, self.Pr_c, self.obsk[:, k_next]) * self.transition_lr
                self.obsk = normalize(self.obsk, axis=1, norm='l1')

            # Update probability of C given S
            self.Pr_c_sv = self.Pr_c_sv + (abs(self.bc - self.Pr_c_sv) / (i + 2))  # + 2 since Pr_c_sv has an element
            self.Pr_c_sv = normalize(self.Pr_c_sv, axis=1, norm='l1')
            # self.WPr_c_sv += self.bc
            # self.Pr_c_sv = normalize(self.WPr_c_sv, axis=1, norm='l1')

    def store(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def ground(self):
        #  'Ground' mdp
        mdp_ground = copy.deepcopy(self)
        bc_ground_all = []
        obskc_all = []
        Tvc_all = []

        noise = np.random.uniform(0, 0.05)

        for c in range(self.bc.shape[1]):
            Svc = np.where(self.Pr_c_sv[:, c] > 0)[0]
            # Filter object belief state
            bc_ground_all.append(self.bc[Svc, c])

            # Filter object observation function
            obskc = self.obsk[Svc]
            obskc_all.append(obskc)

            # Filter transition function
            Tvc = []
            for op in range(len(self.Tv)):
                Tvcop = self.Tv[op][np.ix_(Svc, Svc)]
                # Add some noise to prevent total disagreement with the observation function during planning,
                # which leads to an undefined belief state (i.e. all zeros)
                Tvcop += noise
                Tvcop = normalize(Tvcop, axis=1, norm='l1')
                Tvc.append(Tvcop)
            Tvc_all.append(Tvc)

        To = copy.deepcopy(self.To)
        for op in range(len(To)):
            # Add some noise to prevent total disagreement with the observation function during planning,
            # which leads to an undefined belief state (i.e. all zeros)
            To[op] += noise
            To[op] = normalize(To[op], axis=1, norm='l1')

        mdp_ground.bc = bc_ground_all
        mdp_ground.obsk = obskc_all
        mdp_ground.Tv = Tvc_all
        mdp_ground.To = To

        return mdp_ground

    def toOOPOMDP(self, goal_states):

        # Ground the OPO-RMDP, to perform efficient planning
        mdp_ground = self.ground()

        # Ground goals
        c_states = [list(np.where(self.Pr_c_sv[:, c] > 0)[0]) for c in range(len(mdp_ground.bc))]
        tmp = copy.deepcopy(goal_states)
        goal_states = []
        for g in tmp:
            filtered_g = []
            for c in range(len(g)):
                sv = g[c]
                filtered_g.append(c_states[c].index(sv))
            goal_states.append(filtered_g)

        # Create a belief state (i.e. histogram) for every object
        oo_hists = dict()
        for c in range(len(mdp_ground.bc)):
            oo_hists[c] = pomdp_py.Histogram({ObjectState(sc): mdp_ground.bc[c][sc]
                                              for sc in range(mdp_ground.bc[c].shape[0])})

        # Create the robot (propositional) belief state
        oo_hists['robby'] = pomdp_py.Histogram({RobotState(so): mdp_ground.bo[so]
                                                for so in range(mdp_ground.bo.shape[0])})

        # Initialize the complete (factorized) belief state
        init_belief = MNISTOOBelief(oo_hists)
        goal_states = [[ObjectState(s) for s in goal_state] for goal_state in goal_states]

        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(mdp_ground.obsc, mdp_ground.operators),
                               MNISTTransitionModel(mdp_ground),
                               MNISTObservationModel(mdp_ground.obsk, mdp_ground.obsc),
                               RewardModel(init_belief, goal_states, mdp_ground.obsc, mdp_ground.clusters,
                                           mdp_ground.Tv, mdp_ground.To))

        return agent, mdp_ground
