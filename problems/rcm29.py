import numpy as np
from pymoo.core.problem import ElementwiseProblem


class RCM29(ElementwiseProblem):
    def __init__(self):
        xl = np.array([0, 0, 0, 0, 0, 0, 0])
        xu = np.array([100, 100, 100, 1, 1, 1, 1])
        super().__init__(n_var=7, n_obj=2, n_constr=9,
                         xl=xl, xu=xu, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.copy()
        x[3:] = np.round(x[3:]).astype(int)
        x[3:] = np.clip(x[3:], 0, 1)

        x1, x2, x3, x4, x5, x6, x7v = x

        f1 = ((1 - x4)**2 + (1 - x5)**2 + (1 - x6)**2 - np.log(1 + x7v)
              + (1 - x1)**2 + (2 - x2)**2 + (3 - x3)**2)
        f2 = (1 - x1)**2 + (2 - x2)**2 + (3 - x3)**2

        g1 = x1 + x2 + x3 + x4 + x5 + x6 - 5
        g2 = x6**3 + x1**2 + x2**2 + x3**2 - 5.5
        g3 = x1 + x4 - 1.2
        g4 = x2 + x5 - 1.8
        g5 = x3 + x6 - 2.5
        g6 = x1 + x7v - 1.2
        g7 = x5**2 + x2**2 - 1.64
        g8 = x6**2 + x3**2 - 4.25
        g9 = x5**2 + x3**2 - 4.64

        out["F"] = np.array([f1, f2])
        out["G"] = np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9])
