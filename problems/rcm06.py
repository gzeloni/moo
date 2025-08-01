import numpy as np
from pymoo.core.problem import Problem


class RCM06(Problem):
    def __init__(self):
        super().__init__(n_var=7, n_obj=2, n_ieq_constr=11, xl=np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0]),
                         xu=np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]))

    def _evaluate(self, X, out, *args, **kwargs):
        x1, x2, x3, x4, x5, x6, x7 = X[:, 0], X[:,
                                                1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6]

        f1 = 0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
        f2 = np.sqrt((745 * x4 / (x2 * x3))**2 + 1.69e7) / (0.1 * x6**3)

        g1 = 27 / (x1 * x2**2 * x3) - 1
        g2 = 397.5 / (x1 * x2**2 * x3**2) - 1
        g3 = 1.93 * x4**3 / (x2 * x3 * x6**4) - 1
        g4 = 1.93 * x5**3 / (x2 * x3 * x7**4) - 1
        g5 = np.sqrt((745 * x4 / (x2 * x3))**2 + 1.69e7) / (0.1 * x6**3) - 1100
        g6 = np.sqrt((745 * x5 / (x2 * x3))**2 + 1.575e8) / (0.1 * x7**3) - 850
        g7 = x2 * x3 - 40
        g8 = 5 - x1 / x2
        g9 = x1 / x2 - 12
        g10 = (1.5 * x6 + 1.9) / x4 - 1
        g11 = (1.1 * x7 + 1.9) / x5 - 1

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack(
            [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])
