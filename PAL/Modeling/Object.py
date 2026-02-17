from dataclasses import dataclass
from typing import List


@dataclass
class Object:
    id: str
    states: List[int]
    type: str = 'object'

    OBJ_TYPE: str = 'object'

    def __str__(self):
        return self.id

