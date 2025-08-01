import numpy as np
from pymoo.core.problem import Problem


class RCM29(Problem):
    def __init__(self):

        super().__init__(n_var=2, n_obj=2, n_ieq_constr=2,
                         xl=np.array([0, 0]), xu=np.array([1, 1]))

    def _evaluate(self, X, out, *args, **kwargs):
        x1, x2 = X[:, 0], X[:, 1]

        f1 = x1 + x2
        f2 = (x1 - 0.5)**2 + (x2 - 0.5)**2

        g1 = x1 + x2 - 1.5
        g2 = -x1 - x2 + 0.5

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])
