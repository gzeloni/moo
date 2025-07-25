import numpy as np
from pymoo.core.problem import ElementwiseProblem


class RCM06(ElementwiseProblem):
    def __init__(self):
        xl = np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0])
        xu = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])
        super().__init__(n_var=7, n_obj=2, n_constr=11, xl=xl, xu=xu,
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.copy()
        x[2] = np.round(x[2])
        x1, x2, x3, x4, x5, x6, x7 = x

        f1 = (0.7854*x1*x2**2*(10*x3**2/3 + 14.933*x3 - 43.0934)
              - 1.508*x1*(x6**2 + x7**2)
              + 7.477*(x6**3 + x7**3)
              + 0.7854*(x4*x6**2 + x5*x7**2))

        f2 = (np.sqrt((745*x4/(x2*x3))**2 + 1.69e7))/(0.1*x6**3)

        g1 = 1/(x1*x2**2*x3) - 1/27
        g2 = 1/(x1*x2**2*x3**2) - 1/397.5
        g3 = x3**4/(x2*x3*x6**4) - 1/1.93
        g4 = x3**5/(x2*x3*x7**4) - 1/1.93
        g5 = x2*x3 - 40
        g6 = x1/x2 - 12
        g7 = -x1/x2 + 5
        g8 = 1.9 - x4 + 1.5*x6
        g9 = 1.9 - x5 + 1.1*x7
        g10 = f2 - 1300
        g11 = (np.sqrt((745*x5/(x2*x3))**2 + 1.575e8)/(0.1*x7**3)) - 110

        out["F"] = [f1, f2]
        out["G"] = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]
