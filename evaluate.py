import os
import numpy as np
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import Hypervolume

from problems import RCM06, RCM13, RCM28, RCM29


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def clip_and_cast(x, lb, ub, integer_idx=None, binary_idx=None):
    x = np.clip(x, lb, ub)
    if integer_idx is not None:
        x[integer_idx] = np.round(x[integer_idx]).astype(int)
    if binary_idx is not None:
        x[binary_idx] = np.round(x[binary_idx]).astype(int)
        x[binary_idx] = np.clip(x[binary_idx], 0, 1)
    return x


def equality_to_inequality(h, eps=1e-6):
    return np.abs(h) - eps


def run_nsga(problem, n_gen=250, pop_size=None, seed=1, algorithm="nsga3"):
    if pop_size is None:
        if problem.n_obj == 2:
            pop_size = 100
        elif problem.n_obj == 3:
            pop_size = 92
        else:
            pop_size = 150

    if algorithm.lower() == "nsga2" or problem.n_obj == 2:
        algo = NSGA2(pop_size=pop_size)
    else:
        ref_dirs = get_reference_directions(
            "das-dennis", problem.n_obj, n_partitions=12)
        pop_size = len(ref_dirs)
        algo = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)

    termination = get_termination("n_gen", n_gen)

    res = minimize(problem,
                   algo,
                   termination,
                   seed=seed,
                   save_history=False,
                   verbose=False)
    return res


def plot_front(F, title, path):
    ensure_dir(os.path.dirname(path))

    if F.shape[1] == 2:
        plt.figure()
        plt.scatter(F[:, 0], F[:, 1])
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    elif F.shape[1] == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(F[:, 0], F[:, 1], F[:, 2])
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    else:

        plt.figure()
        for i in range(F.shape[0]):
            plt.plot(range(F.shape[1]), F[i, :], alpha=0.2)
        plt.xlabel("Objective index")
        plt.ylabel("Value")
        plt.title(title + " (parallel coords)")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def save_population(res, path):
    ensure_dir(os.path.dirname(path))
    np.savez(path, X=res.X, F=res.F, CV=res.CV)


def compute_hv(F, ref_point=None):
    F = np.asarray(F, dtype=float)
    bad = ~np.isfinite(F).all(axis=1)
    F = F[~bad]
    if F.size == 0:
        return np.nan

    if ref_point is None:
        ref_point = 1.1 * np.max(F, axis=0)

    hv = Hypervolume(ref_point=ref_point)
    return hv.do(F)


def experiment(problem_cls, label, n_gen=300, seeds=(1, 2, 3)):
    results = []
    all_fronts = []
    for s in seeds:
        res = run_nsga(problem_cls(), n_gen=n_gen, seed=s)
        results.append(res)
        all_fronts.append(res.F)
        save_population(res, f"results/{label}_seed{s}.npz")

    F = np.vstack(all_fronts)

    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    F_nd = F[nds]

    hv = compute_hv(F_nd)
    plot_front(F_nd, f"{label} (combined, HV={hv:.3e})",
               f"results/{label}_PF.png")

    print(f"{label}: Combined ND solutions = {len(F_nd)}, HV={hv:.5e}")


def main():
    ensure_dir("results")
    configs = [
        (RCM06, "RCM06", 300),
        (RCM13, "RCM13", 400),
        (RCM28, "RCM28", 300),
        (RCM29, "RCM29", 300),
    ]

    for cls, label, n_gen in configs:
        experiment(cls, label, n_gen=n_gen, seeds=(1, 11, 21))


if __name__ == "__main__":
    main()
