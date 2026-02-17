import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List
import numpy as np
from PIL import Image


@dataclass
class ClusteringGT:

    cluster_labels: List[int] = field(default_factory=list)
    features: list = field(default_factory=list)
    object_states: list = field(default_factory=list)
    # DEBUG
    all_digits: set = field(default_factory=set)
    gt_states: list = field(default_factory=list)
    digit_2_labels: defaultdict = field(default_factory=lambda: defaultdict(set))
    labels_2_digit: defaultdict = field(default_factory=lambda: defaultdict(str))
    cluster_noise: int = 0

    def cluster(self, dataset):

        for i in range(len(self.cluster_labels), len(dataset)):
            gt_digit = dataset['eval'][i]['digit']
            new_obj_state = {f"{gt_digit.split('-')[0]}_{k}": v
                             for k, v in dataset['eval'][i]['state']['objects'][gt_digit].items()
                             if k != 'pos'}

            if frozenset(new_obj_state.items()) not in self.object_states:
                self.object_states.append(frozenset(new_obj_state.items()))
                self.cluster_labels.append(len(self.object_states) - 1)
                self.all_digits.add(dataset['eval'][i]['digit'])
                self.digit_2_labels[dataset['eval'][i]['digit']].add(len(self.object_states) - 1)
                self.labels_2_digit[len(self.object_states) - 1] = dataset['eval'][i]['digit']
            else:
                self.cluster_labels.append(self.object_states.index(frozenset(new_obj_state.items())))

            self.gt_states.append(new_obj_state)
            self.features.append(np.array(Image.open(dataset['rgb'][i]).convert('L').histogram()))

        cluster_labels_noisy = []
        for i, l in enumerate(self.cluster_labels):
            if random.random() < self.cluster_noise and l in cluster_labels_noisy:  # Ensure noisy clustering does not fully remove clusters

                prev_digit_clusters = self.digit_2_labels[self.labels_2_digit[l]].intersection(set(self.cluster_labels[:i + 1]))
                cluster_labels_noisy.append(random.choice(list(prev_digit_clusters)))
            else:
                cluster_labels_noisy.append(l)
        cluster_labels = cluster_labels_noisy

        return np.array(cluster_labels)

    def store(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def predict(self, samples):
        preds = []
        for sample in samples:
            gt_digit = sample['eval']['digit']

            new_obj_state = {f"{gt_digit.split('-')[0]}_{k}": v
                             for k, v in sample['eval']['state']['objects'][gt_digit].items()
                             if k != 'pos'}

            if frozenset(new_obj_state.items()) not in self.object_states:
                predicted_cluster = None
                preds.append(predicted_cluster)
            else:
                predicted_cluster = self.object_states.index(frozenset(new_obj_state.items()))
                self.cluster_labels.append(predicted_cluster)
                preds.append(f"c{predicted_cluster}")
        return preds

