import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import pomdp_py

from pomdp_py import TreeDebugger
from sklearn.preprocessing import normalize

from PAL.Modeling.Action import Action
from PAL.Clustering.ClusteringGT import ClusteringGT
from PAL.Modeling.Model import Model
from PAL.Clustering.Cluster import Cluster
from PAL.Modeling.Operator import Operator
from PAL.Modeling.Trace import Trace
from utils.Logger import Logger
from utils.oopomdp import MNISTOOObservation, RobotState
from utils.config import get_ag_cfg, get_run_cfg

import gymnasium as gym

from utils.util import exact_covering


@dataclass
class OracleAgent:

    logger: Logger = field(default_factory=Logger)
    dataset: pd.DataFrame = field(default_factory=pd.DataFrame)
    trace: Trace = field(default_factory=Trace)
    mdp: Model = field(init=False)
    clustering: ClusteringGT = field(default_factory=ClusteringGT)
    prev_clusters: Dict[str, Cluster] = None
    model_path: str = None
    cfg: Dict[str, Any] = field(default_factory=get_ag_cfg)
    run_cfg: Dict[str, Any] = field(default_factory=get_run_cfg)  # TODO: something else


    def discretize(self, obs: Any) -> int:
        """
        Discretize the input observation. For example, the input observation may be an RGB image and the
        output (discretized) observation an integer ID.
        """
        repr_obs = repr(obs)
        if repr_obs not in self.mdp.discrete_obs:
            self.mdp.discrete_obs[repr_obs] = len(self.mdp.discrete_obs)
        return self.mdp.discrete_obs[repr_obs]

    def learn(self,
              env_id: str,
              env_kwargs: Dict[str, Any],
              input_hyperparams_path: str = None,
              load_checkpoint: bool = False,
              seed: int = 123) -> None:

        # Set log directory
        self.logger.set_log_dir(f'res/{env_id}/{self.cfg["method"]}/train')

        # Clean previous runs
        self.clustering = ClusteringGT()
        self.dataset = pd.DataFrame()
        self.trace = Trace()

        # Init env
        env = gym.make(env_id, **env_kwargs)
        env = env.unwrapped

        ndigits = len(env._state['objects'])

        self.mdp = Model(env.operators)

        # Ground truth transition functions for nullary operators
        self.mdp.To = [np.zeros((ndigits, ndigits)) for _ in range(len(self.mdp.ops_zeroary))]
        # noop
        self.mdp.To[0] = np.zeros((ndigits, ndigits))

        right = np.roll(np.eye(ndigits), 1)[::-1].T[::-1]
        left = np.roll(np.eye(ndigits), 1).T
        noop = np.eye(ndigits)
        stop = np.eye(ndigits)
        # Possibly add actuation noise
        if env.cfg['actuation_noise'] > 0:
            right = (np.clip(right - env.cfg['actuation_noise'], 0, 1)
                     + np.identity(right.shape[0]) * env.cfg['actuation_noise'])
            left = (np.clip(left - env.cfg['actuation_noise'], 0, 1)
                    + np.identity(left.shape[0]) * env.cfg['actuation_noise'])

        # stop
        self.mdp.To[0] = stop
        # right
        self.mdp.To[1] = right
        # left
        self.mdp.To[2] = left

        # Ground truth object observation function for propositional state
        self.mdp.obsc = np.eye(ndigits)

        # Sort discrete observation by position to align with the rotate/right transition functions
        for c, d in enumerate(dict(sorted(env._state['objects'].items(), key=lambda x: x[1]['pos']))):
            for flipped in [0, 1]:
                for rotation in [0, 180]:
                    obs = env.digits[d][flipped][rotation]

                    self.discretize(obs)

        # Ground truth object observation function for clusters
        self.mdp.obsk = np.zeros((len(self.mdp.discrete_obs), len(self.mdp.discrete_obs)))
        Sc = 4
        self.mdp.Pr_c_sv = np.zeros((len(self.mdp.discrete_obs), ndigits))
        dtypes = [d.split('-')[0] for d in env._state['objects']]
        types = list()
        for d in dtypes:
            if d not in types:
                types.append(d)

        for c, d in enumerate(dict(sorted(env._state['objects'].items(), key=lambda x: x[1]['pos']))):
            sc = 0
            for flipped in [0, 1]:
                for rotation in [0, 180]:
                    obs = env.digits[d][flipped][rotation]
                    k_id = self.discretize(obs)

                    sv = types.index(d.split('-')[0]) * Sc + sc
                    self.mdp.obsk[sv, k_id] += 1.
                    sc += 1

                    # Set the probability Pr(c | sv) of size Sv x C
                    self.mdp.Pr_c_sv[sv, c] = 1.

        self.mdp.Pr_c_sv = normalize(self.mdp.Pr_c_sv, axis=1, norm='l1')
        self.mdp.obsk = normalize(self.mdp.obsk, axis=1, norm='l1')

        # Ground truth transition functions for unary operators
        self.mdp.Tv = [np.zeros((self.mdp.obsk.shape[0], self.mdp.obsk.shape[0]))
                       for _ in range(len(self.mdp.ops_unary))]

        # flip and rotate for a single object
        rotate = np.roll(np.eye(Sc), 2, axis=1)[::-1]
        flip = np.roll(np.eye(Sc), 2, axis=1)

        # Possibly add actuation noise
        if env.cfg['actuation_noise'] > 0:
            rotate = (np.clip(rotate - env.cfg['actuation_noise'], 0, 1)
                      + np.identity(rotate.shape[0]) * env.cfg['actuation_noise'])
            flip = (np.clip(flip - env.cfg['actuation_noise'], 0, 1)
                    + np.identity(flip.shape[0]) * env.cfg['actuation_noise'])

        for i in range(self.mdp.obsk.shape[0] // Sc):
            self.mdp.Tv[0][Sc * i: Sc * (i + 1), Sc * i: Sc * (i + 1)] = rotate
            self.mdp.Tv[1][Sc * i: Sc * (i + 1), Sc * i: Sc * (i + 1)] = flip

        # Set transition matrix weights
        self.mdp.WTv = np.ones_like(self.mdp.Tv)
        self.mdp.WTo = np.ones_like(self.mdp.To)

        # Store the updated OPO-RMDP and clustering
        self.mdp.store(f"{self.logger.models_dir}/model.pkl")
        self.clustering.store(f"{self.logger.models_dir}/clustering.pkl")

    def load_model(self, path: str) -> None:
        self.model_path = path
        with open(path, 'rb') as f:
            self.mdp = pickle.load(f)

    def load_clustering(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.clustering = pickle.load(f)

    def load_checkpoint(self, path: str) -> None:
        self.load_model(f"{path}/model.pkl")
        self.load_clustering(f"{path}/clustering.pkl")

    def cluster_rgbs(self, dataset):

        cluster_labels = self.clustering.cluster(dataset)

        rgb2clusters = dict()
        self.mdp.clusters = dict()
        for l in set(cluster_labels):
            idx = np.where(cluster_labels == l)[0]
            c = Cluster(cluster_id=f"c{l}",
                        features=[self.clustering.features[i] for i in idx],
                        imgs=[dataset['rgb'][i] for i in idx],
                        gt_states=[self.clustering.gt_states[i] for i in idx])
            for i in idx:
                rgb2clusters[dataset['rgb'][i]] = c.id
            self.mdp.clusters[c.id] = c

        return rgb2clusters

    # Currently assume the rgb is already the bbox image of a single object
    def detect_objects(self, rgb):
        return [np.array(rgb)]

    def clusters2states(self, cluster_ids: List[str]) -> np.ndarray:
        SvC = list()

        for k_id in cluster_ids:

            if k_id is None:
                return np.array([None])

            Svc_k = []
            Pr_c = np.einsum('s, sc -> c', self.mdp.obsk[:, k_id], self.mdp.Pr_c_sv)

            for c in np.nonzero(Pr_c)[0]:
                Sc, = np.where(self.mdp.obsk[:, k_id] > .5)
                [Svc_k.append((c, sc)) for sc in Sc]
            SvC.append(Svc_k)

        goal_states = []
        for goal_state in exact_covering(SvC):
            goal_states.append([sc for c, sc in sorted(goal_state, key=lambda x: x[0])])

        return np.array(goal_states)


    def solve(self,
              env_id: str,
              env_kwargs: Dict[str, Any],
              input_model_path: str,
              seed: int = 123,
              max_steps: int = 100) -> None:

        # Set log directories
        self.logger.set_log_dir(f"res/{env_id}/{self.cfg['method']}/test")
        start = datetime.now()

        # Initialize model
        assert input_model_path is not None
        self.load_checkpoint(input_model_path)

        # Initialize trace
        self.trace = Trace()

        # Perceive initial state
        env = gym.make(env_id, **env_kwargs)
        env = env.unwrapped
        env.max_steps = max_steps
        obs, _ = env.reset(options={'_state': env._state, 'goal_states': env.goal_states})

        # Get goal observations
        goal_obs = env.get_goal_obs()

        # Discretize goal observations
        goal_clusters = [self.mdp.discrete_obs[repr(o)]
                         if repr(o) in self.mdp.discrete_obs else None
                         for o in goal_obs]

        # Map goal observations to goal states
        goal_states = self.clusters2states(goal_clusters)
        current_cluster = self.mdp.discrete_obs[repr(obs)] if repr(obs) in self.mdp.discrete_obs else None
        obs = self.discretize(obs)
        self.trace.observations.append(obs)

        optimal_cost = env.distance_to_success()
        metrics = pd.DataFrame()
        cumulative_reward = 0

        if current_cluster is None or np.any([g is None for g in goal_states]):
            if current_cluster is None:
                self.logger.write('[Failure] initial agent state is an unseen cluster/object state, deeming'
                                  'the goal unfeasible')
            else:
                self.logger.write(f'[Failure] Goal deemed unfeasible since it involves an unseen cluster/object state. '
                                  f'Number of learned clusters {self.mdp.obsk.shape[0]}.')
            i = 0
            done = False

        else:
            self.mdp.set_current_state(current_cluster)

            ooagent, mdp_ground = self.mdp.toOOPOMDP(goal_states)

            if self.cfg['planner'] == 'POUCT':
                self.planner = pomdp_py.POUCT(rollout_policy=ooagent.policy_model,
                                              num_sims=500,
                                              exploration_const=50000 * len(mdp_ground.bc),
                                              # exploration_const=100 * len(mdp_ground.bc),
                                              discount_factor=0,
                                              max_depth=0
                                              )
            elif self.cfg['planner'] == 'POMCP':
                ooagent.set_belief(
                    pomdp_py.Particles.from_histogram(ooagent.cur_belief, num_particles=100),
                    prior=True
                )
                self.planner = pomdp_py.POMCP(rollout_policy=ooagent.policy_model)
            else:
                raise NotImplementedError

            for i in range(max_steps):

                # Check goal feasibility: a goal is not feasible if the number of goal objects is higher than
                # the number of objects learned by the agent
                if len(goal_states) == 0:
                    self.logger.write(f'[Failure] Goal deemed unfeasible. Number of learned objects {self.mdp.bc.shape[1]}.'
                                      f'Number of goal objects {len(goal_clusters)}.')
                    action = Action(name='stop', types=[], objects=[])
                else:
                    # Plan
                    action = self.planner.plan(ooagent)
                    # dd = TreeDebugger(ooagent.tree)
                    # dd.pp
                    # dd.mbp

                a_name = action.name
                objects = []
                for obj_id in action.objects:
                    for obj in mdp_ground.objects:
                        if obj.id == f"obj_{obj_id}":
                            objects.append({obj.id: [c.mean() for c in obj.states]})

                # Act and perceive
                # TODO: use different ids for different objects
                action_id = env.operators.index(Operator(action.name, action.types))
                obs, reward, done, truncated, info = env.step(action_id)

                self.logger.write(f'{i}: {action}')

                if done:
                    print('[Debug] GOAL ACHIEVED')

                # Create a new observation from the current perceptions, i.e. a set of perceptions (one for each sensor)
                new_obs_cluster = self.mdp.discrete_obs[repr(obs)] if repr(obs) in self.mdp.discrete_obs else None
                obs = self.discretize(obs)
                self.trace.observations.append(obs)

                # Check goal feasibility: a goal is not feasible if the observed object state has not been learned,
                # i.e., is not in the transition function of that object.
                if new_obs_cluster is None:
                    self.logger.write(f'[Failure] Goal deemed unfeasible. The agent is observing an unseen '
                                      f'cluster/object state. Number of learned clusters {self.mdp.obsk.shape[0]}.')
                    action = Action(name='stop', types=[], objects=[])
                    a_name = action.name
                    self.trace.actions.append(action)
                else:
                    # Update trace actions
                    operator = env.operators[action_id]
                    self.trace.actions.append(Action(objects=operator.types, name=operator.name, types=operator.types))

                # Update the agent belief. If the planner is POMCP, planner.update
                # also automatically updates agent belief.
                if self.cfg['planner'] != 'POUCT':
                    raise NotImplementedError

                if self.cfg['planner'] == 'POUCT' and action.name != 'stop':

                    # Update model with the last action and observation
                    self.mdp.update(self.trace, update_transf=False)
                    self.trace.reset()

                    mdp_ground = self.mdp.ground()
                    Pr_c = np.einsum('s, sc -> c', mdp_ground.bo, mdp_ground.obsc)
                    Pr_k_bc = np.array([np.einsum('s, s -> ', mdp_ground.obsk[c][:, new_obs_cluster],
                                                  mdp_ground.bc[c])
                                        for c in range(len(mdp_ground.bc))])

                    real_observation = MNISTOOObservation({c: new_obs_cluster
                                                           for c, Pr_k_c in enumerate(Pr_k_bc) if Pr_k_c * Pr_c[c] > 0})
                    assert len(real_observation.ktuple) > 0, ('Check and possibly lower threshold or add condition to'
                                                              'take minimum objid when there is no probability '
                                                              'Pr_k_c above the threshold')

                    # ooagent._history = None  # TODO: truncate history ?
                    ooagent.update_history(action, real_observation)
                    self.planner.update(ooagent, action, real_observation)

                    # Update every object belief state
                    for objid in ooagent.cur_belief.object_beliefs:
                        belief_obj = ooagent.cur_belief.object_belief(objid)
                        if isinstance(belief_obj, pomdp_py.Histogram):

                            if objid == 'robby':
                                if len(action.objects) == 0 and action.name != 'noop':
                                    new_belief = pomdp_py.Histogram({RobotState(i): mdp_ground.bo[i]
                                                                     for i in range(mdp_ground.bo.shape[0])})
                                    ooagent.cur_belief.set_object_belief(objid, new_belief)
                            else:

                                # If a new object state has been discovered, fully update the oopomdp
                                if len(mdp_ground.bc[objid]) != len(belief_obj):
                                    ooagent, mdp_ground = self.mdp.toOOPOMDP(goal_states)
                                else:
                                    for sc in belief_obj.histogram.keys():
                                        belief_obj[sc] = mdp_ground.bc[objid][sc.sc]

                                    assert 1 - 1e-5 < sum(belief_obj.histogram.values()) < 1 + 1e5, 'Newly check if object belief should be normalized'

                                    ooagent.cur_belief.set_object_belief(objid, belief_obj)

                        else:
                            print('Object belief update must be customized as done for POUCT to take into account'
                                  'the probability of observing an object, i.e., probabilistic observations')
                            raise NotImplementedError

                if (i + 1) % 10 == 0 or i == 0 or a_name == 'stop':
                    # TODO: deal with this 'special' case where the initial state is already a goal one
                    if optimal_cost == 0:
                        spl = float(done) * ((optimal_cost + 1) / max(i + 1, (optimal_cost + 1)))
                    else:
                        spl = float(done) * (optimal_cost / max(i, optimal_cost))
                    evaluation = {
                        'Objects': len(env._state['objects']),
                        'Non determinism': env.cfg['actuation_noise'],
                        'Cumulative reward': cumulative_reward,
                        'Success rate': float(done),
                        'Distance to success': env.distance_to_success(),
                        # 'SPL': float(done) * (optimal_cost / max(i, optimal_cost)),
                        'SPL': spl,
                        'Time seconds': (datetime.now() - start).seconds,
                        'Steps': i + 1,
                        'Max steps': env.max_steps,
                        'Method': self.cfg['planner'],
                        'Seed': env.seed,
                        'Model path': self.model_path
                    }
                    metrics = pd.concat([metrics, pd.DataFrame([evaluation])], ignore_index=True)

                    if a_name == 'stop':
                        break

        # TODO: deal with this 'special' case where the initial state is already a goal one
        if optimal_cost == 0:
            spl = float(done) * ((optimal_cost + 1) / max(i + 1, (optimal_cost + 1)))
        else:
            spl = float(done) * (optimal_cost / max(i, optimal_cost))

        evaluation = {
            'Objects': len(env._state['objects']),
            'Non determinism': env.cfg['actuation_noise'],
            'Cumulative reward': cumulative_reward,
            'Success rate': float(done),
            'Distance to success': env.distance_to_success(),
            'SPL': spl,
            'Time seconds': (datetime.now() - start).seconds,
            'Steps': i + 1,
            'Max steps': max_steps,
            'Method': self.cfg['planner'],
            'Seed': env.seed,
            'Model path': self.model_path
        }
        metrics = pd.concat([metrics, pd.DataFrame([evaluation])], ignore_index=True)

        self.logger.write(f'================== METRICS ==================')
        self.logger.write(f'Success rate:{float(done)}')
        self.logger.write(f'Distance to success:{env.distance_to_success():.2}')
        self.logger.write(f'SPL:{spl:.2}')
        self.logger.write(f'Steps:{i + 1}')
        self.logger.write(f'Time seconds:{(datetime.now() - start).seconds}')

        metrics.to_excel(f"{self.logger.log_dir}/metrics.xlsx", index=False, float_format="%.2f")

        # Clean image files
        shutil.rmtree(self.logger.img_dir)

        # Close environment
        env.close()
