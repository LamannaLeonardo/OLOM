from typing import List, Any

from PAL.Modeling.Operator import Operator


class Action(Operator):
    """
    An action is a grounded operator, e.g. operator = pickup(x - block) and action = pickup(block0)
    """

    def __init__(self, objects: List[Any] = list, **kwargs):
        super(Action, self).__init__(**kwargs)
        self.objects = objects

    def __str__(self):
        return f"{self.name}({','.join(self.objects)})"

    def __eq__(self, other):
        if isinstance(other, Operator) and not isinstance(other, Action):
            return super().__eq__(other)
        return super().__eq__(other) and self.objects == other.objects




