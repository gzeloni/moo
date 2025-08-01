import os
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from problems import RCM06, RCM13, RCM28, RCM29


def evaluate(problems, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    results = []

    for problem in problems:
        name = problem.__class__.__name__
        algorithm = NSGA2(pop_size=100)
        res = minimize(problem, algorithm, ('n_gen', 200),
                       seed=1, verbose=False)
        F = res.F
        min_F = np.min(F, axis=0)
        max_F = np.max(F, axis=0)
        F_norm = (F - min_F) / (max_F - min_F)

        ref_point = np.array([1.1, 1.1])
        hv = HV(ref_point=ref_point)(F_norm)

        plt.figure()
        plt.scatter(F[:, 0], F[:, 1], c='blue', s=20)
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title(f'Pareto Front - {name}')
        plot_path = os.path.join(save_dir, f'{name}.png')
        plt.savefig(plot_path)
        plt.close()

        results.append({
            'name': name,
            'hv': hv,
            'plot_path': plot_path
        })
        print(f"{name} - HV: {hv:.4f}")

    return results


if __name__ == "__main":
    problems = [RCM06(), RCM13(), RCM28(), RCM29()]
    results = evaluate(problems)
    for result in results:
        print(result)
