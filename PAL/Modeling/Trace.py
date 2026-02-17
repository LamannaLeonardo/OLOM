from dataclasses import dataclass, field
from typing import List

from PAL.Modeling.Action import Action
from PAL.Modeling.Observation import Observation


@dataclass
class Trace:

    observations: List[Observation] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)

    def reset(self):
        self.actions = list()
        self.observations = self.observations[-1:]
