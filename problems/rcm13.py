import numpy as np
from pymoo.core.problem import Problem


class RCM13(Problem):
    def __init__(self):

        super().__init__(n_var=4, n_obj=2, n_ieq_constr=5, xl=np.array(
            [12, 12, 12, 12]), xu=np.array([60, 60, 60, 60]))

    def _evaluate(self, X, out, *args, **kwargs):
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        f1 = ((1/6.931) - (x1 * x2) / (x3 * x4))**2
        f2 = x1 + x2 + x3 + x4

        g1 = x1 - x2
        g2 = x2 - x3
        g3 = x3 - x4
        g4 = 20 - (x1 + x2 + x3 + x4)
        g5 = (x1 * x2) / (x3 * x4) - 0.5

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2, g3, g4, g5])
