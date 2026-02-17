import numpy as np
import pandas as pd
from PIL import Image


class Cluster:

    def __init__(self, cluster_id, features, imgs, gt_states):
        self.id = cluster_id
        self.features = np.array(features)
        self.imgs = imgs

        # DEBUG
        eval = pd.DataFrame(data=pd.DataFrame(gt_states).value_counts(dropna=False, normalize=False))
        state_vars = eval.to_dict(orient='tight')['index_names']
        self.eval = []
        for el in eval.iterrows():
            obj_state = dict(zip(state_vars, el[0]))
            self.eval.append((obj_state, el[1]['count']))

    def __eq__(self, other):
        return self.id == other.svec

    def __gt__(self, other):
        return int(self.id[1:]) > int(other.svec[1:])

    def __hash__(self):
        return hash(self.id)

    def mean(self):
        return self.features.mean(axis=0).round(4)

    def sample(self):
        return np.array(Image.open(np.random.choice(self.imgs)))
