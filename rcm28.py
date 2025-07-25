import numpy as np
from pymoo.core.problem import ElementwiseProblem


def equality_to_inequality(h, eps=1e-6):
    return np.abs(h) - eps


class RCM28(ElementwiseProblem):
    def __init__(self):
        xl = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        xu = np.array([100, 100, 100, 100, 100, 100, 1, 1])
        super().__init__(n_var=8, n_obj=2, n_constr=9,
                         xl=xl, xu=xu, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        x = x.copy()
        x[6] = int(round(x[6]))
        x[7] = int(round(x[7]))

        x1, x2, x3, x4, x5, x6, x7b, x8b = x

        f1 = 7.5 * x7b + 5.5 * x8b + 7 * x5 + 6 * x6 + 5 * (x1 + x2)
        f2 = x1 + x2

        h1 = x7b + x8b - 1
        h2 = x3 - 0.9 * (1 - np.exp(0.5 * x5)) * x1
        h3 = x4 - 0.8 * (1 - np.exp(0.4 * x6)) * x2
        h4 = x3 + x4 - 10
        h5 = x3 * x7b + x4 * x8b - 10

        g1 = x5 - 10 * x7b
        g2 = x6 - 10 * x8b
        g3 = x1 - 20 * x7b
        g4 = x2 - 20 * x8b

        eps = 1e-6
        G = np.array([
            g1, g2, g3, g4,
            equality_to_inequality(h1, eps),
            equality_to_inequality(h2, eps),
            equality_to_inequality(h3, eps),
            equality_to_inequality(h4, eps),
            equality_to_inequality(h5, eps),
        ])

        out["F"] = np.array([f1, f2])
        out["G"] = G
