from typing import List, Tuple

from pysat.formula import CNF
from pysat.solvers import Solver


def exact_covering(list_cover: List[List[Tuple]]) -> List[List[Tuple]]:
    """
    Takes as input a list of sublists where each element is a tuple (idx, v), and compute the list
    of tuple combinations such that exaclty one tuple is chosen from every sublist and the set of all
    tuple indices is fully covered.
    For example:
    list1 = [
    [(0, a), (3, a)],
    [(1, b), (2, b)],
    [(1, c), (2, c)],
    [(0, c), (3, d)]
    ],
    and the list of valid combinations is:
    [
    [(0, a), (1, b), (2, c), (3, d)],
    [(0, a), (1, c), (2, b), (3, d)],
    [(0, c), (1, b), (2, c), (3, a)],
    [(0, c), (1, c), (2, b), (3, a)]
    ]
    In particular, the exact-covering problem is formulated as a SAT problem.
    """

    # Build CNF
    cnf = CNF()
    vars_map = {}
    var_id = 1
    indices = {el[0] for sublst in list_cover for el in sublst}
    # Map each choice to a SAT var
    for j, sub in enumerate(list_cover):
        # at least one: (X_j0 ∨ X_j1 ∨ ...)
        clause = []
        for k in range(len(sub)):
            vars_map[(j, k)] = var_id
            clause.append(var_id)
            var_id += 1
        cnf.append(clause)
        # at most one: pairwise ¬X_jp ∨ ¬X_jq
        for p in range(len(sub)):
            for q in range(p + 1, len(sub)):
                cnf.append([-vars_map[(j, p)], -vars_map[(j, q)]])

    # each index exactly once
    for i in indices:
        lits = [vars_map[(j, k)]
                for j, sub in enumerate(list_cover)
                for k, (idx, _) in enumerate(sub) if idx == i]
        # at least one
        cnf.append(lits)
        # at most one
        for p in range(len(lits)):
            for q in range(p + 1, len(lits)):
                cnf.append([-lits[p], -lits[q]])

    # Enumerate models
    solutions = []
    with Solver(bootstrap_with=cnf.clauses) as s:
        for model in s.enum_models():
            sel = [list_cover[j][k] for (j, k), vid in vars_map.items() if model[vid - 1] > 0]
            solutions.append(sel)

    # print("[Debug] All solutions via SAT:", solutions)
    return solutions
