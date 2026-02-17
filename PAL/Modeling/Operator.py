from dataclasses import dataclass, field
from functools import cached_property
from typing import List


@dataclass
class Operator:

    name: str
    types: List[str] = field(default_factory=list)

    @cached_property
    def arity(self):
        return len(self.types)

    def __str__(self):
        return f"{self.name}({','.join(self.types)})"

    def __eq__(self, other):
        assert isinstance(other, Operator)
        return self.name == other.name and self.types == other.types
