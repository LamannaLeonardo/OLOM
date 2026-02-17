from dataclasses import dataclass
from typing import List


@dataclass
class Observation:

    id: int
    perceptions: List[str]
    clusters: List[str] = None
    gt_loc: int = None

    def __hash__(self):
        return hash(self.id)




        



