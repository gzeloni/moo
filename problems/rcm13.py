import numpy as np
from pymoo.core.problem import ElementwiseProblem


class RCM13(ElementwiseProblem):
    def __init__(self):
        xl = np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0])
        xu = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])
        super().__init__(n_var=7, n_obj=3, n_constr=11,
                         xl=xl, xu=xu, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.copy()
        x[2] = np.round(x[2])
        x1, x2, x3, x4, x5, x6, x7 = x

        f1 = (0.7854 * x2**2 * x1 * (14.9334 / x3 - 43.0934 + 3.3333 * x3**2)
              + 0.7854 * (x5 * x7**2 + x4 * x6**2)
              - 1.508 * x1 * (x7**2 + x6**2) + 7.477 * (x7**3 + x6**3))

        f2 = 10 * x6**-3 * np.sqrt(16.91e6 + (745 * x4 / (x2 * x3))**2)
        f3 = 10 * x7**-3 * np.sqrt(157.5e6 + (745 * x5 / (x2 * x3))**2)

        g1 = 1/(x1 * x2**2 * x3) - 1/27
        g2 = 1/(x1 * x2**2 * x3**2) - 1/397.5
        g3 = 1/(x2 * x6**4 * x3 * x4**-3) - 1/1.93
        g4 = 1/(x2 * x7**4 * x3 * x5**-3) - 1/1.93
        g5 = 10 * x6**-3 * np.sqrt(16.91e6 + (745 * x4 / (x2 * x3))**2) - 1100
        g6 = 10 * x7**-3 * np.sqrt(157.5e6 + (745 * x5 / (x2 * x3))**2) - 850
        g7 = x2 * x3 - 40
        g8 = -x1 / x2 + 5
        g9 = x1 / x2 - 12
        g10 = 1.5 * x6 - x4 + 1.9
        g11 = 1.1 * x7 - x5 + 1.9

        out["F"] = np.array([f1, f2, f3])
        out["G"] = np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])
