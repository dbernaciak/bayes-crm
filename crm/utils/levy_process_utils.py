"""Utilities for Levy processes."""

import itertools
import inspect
import json
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import time
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import integrate
from .general_utils import hash_args
from .numerics import arrival_times
from ..crm_approx import ApproxProcess, envelope
from ..fk import ferguson_klass
from ..levy_processes import (
    beta_process,
    g_beta_process,
    stable_beta_process,
    g_stable_beta_process,
)


def measure_time_approx_process(num_fits, *args, size=100, **kwargs):
    start_time = time.time()
    for _ in tqdm(range(num_fits)):
        p = ApproxProcess(*args, **kwargs)
        p.PROCESS_MIN = 1e-100
        times_of_arrival = arrival_times(size)
        _ = p.generate(times_of_arrival)
    return (time.time() - start_time) / num_fits


def process_errors_and_jump_sizes(
    num_fits: int,
    process: Callable,
    params: dict,
    n_grids: list[int],
    g_process: Callable,
    bounds=(0, 1),
    use_trap: bool = False,
    thr: float = 0.5,
    n_jumps=100,
    cache: bool = False,
) -> Tuple[dict, dict]:
    """Measure errors and jump sizes for a given process.

    Args:
        num_fits (int): Number of fits.
        process (Callable): Levy process.
        params (dict): Parameters for the process.
        n_grids (list[int]): Number of grids.
        g_process (Callable): Non-exponential part of the process.
        bounds (Tuple[float, float], optional): Bounds for the integral. Default is (1e-10, 1).
        use_trap (bool): Use trapezoidal method.
        thr (float): Threshold.
        n_jumps (int): Number of jumps.

    Tuple[dict, dict]: A tuple containing two dictionaries. The first dictionary contains the errors for each grid size. The second dictionary contains the jump sizes for each grid size.
    """

    if cache:
        hash_hex = hash_args(
            num_fits,
            process.__name__,
            params,
            n_grids,
            g_process.__name__,
            bounds,
            use_trap,
            thr,
            n_jumps,
        )
        try:
            with open(
                f"./data/cache/{inspect.currentframe().f_code.co_name}_{hash_hex}_1.json",
                "r",
            ) as f:
                errors_dec = json.load(f)
            with open(
                f"./data/cache/{inspect.currentframe().f_code.co_name}_{hash_hex}_2.json",
                "r",
            ) as f:
                jump_sizes = json.load(f)

            with open(
                f"./data/cache/{inspect.currentframe().f_code.co_name}_{hash_hex}_3.json",
                "r",
            ) as f:
                errors_nodec = json.load(f)

            with open(
                f"./data/cache/{inspect.currentframe().f_code.co_name}_{hash_hex}_4.json",
                "r",
            ) as f:
                errors_approx = json.load(f)
            return errors_dec, errors_nodec, errors_approx, jump_sizes
        except FileNotFoundError:
            pass

    jump_sizes = {str(ng): [] for ng in n_grids}
    errors_dec = {str(ng): [] for ng in n_grids}
    errors_nodec = {str(ng): [] for ng in n_grids}
    errors_approx = {str(ng): [] for ng in n_grids}
    kappa = -1
    if "sigma" in params.keys():
        kappa = -1 - params["sigma"]
    if use_trap:
        kappa = None

    for ng in n_grids:
        for _ in tqdm(range(num_fits)):
            times_of_arrival = np.random.exponential(size=n_jumps).cumsum()
            fk = ferguson_klass(
                times_of_arrival, process(**params), upper_lim=bounds[1]
            )
            idx = np.argwhere(fk != 0)
            bp = ApproxProcess(
                process(**params),
                ng,
                g_process(**params) if not use_trap else None,
                kappa,
                bounds=bounds,
                thr=thr,
            )
            bp_nodec = ApproxProcess(
                process(**params),
                ng,  # int(((ng - 1) - 1) / 2 + 1),
                None,
                None,
                bounds=bounds,
                thr=thr,
            )
            num = bp.generate(times_of_arrival)
            num_nodec = bp_nodec.generate(times_of_arrival)
            errors_dec[str(ng)].extend((abs(num[idx] - fk[idx]) / fk[idx]).ravel())
            errors_nodec[str(ng)].extend(
                (abs(num_nodec[idx] - fk[idx]) / fk[idx]).ravel()
            )
            errors_approx[str(ng)].extend(
                (abs(num[idx] - num_nodec[idx]) / fk[idx]).ravel()
            )
            jump_sizes[str(ng)].extend(fk[idx].ravel())

    if cache:
        with open(
            f"./data/cache/{inspect.currentframe().f_code.co_name}_{hash_hex}_1.json",
            "w",
        ) as f:
            json.dump(errors_dec, f)
        with open(
            f"./data/cache/{inspect.currentframe().f_code.co_name}_{hash_hex}_2.json",
            "w",
        ) as f:
            json.dump(jump_sizes, f)
        with open(
            f"./data/cache/{inspect.currentframe().f_code.co_name}_{hash_hex}_3.json",
            "w",
        ) as f:
            json.dump(errors_nodec, f)
        with open(
            f"./data/cache/{inspect.currentframe().f_code.co_name}_{hash_hex}_2.json",
            "w",
        ) as f:
            json.dump(errors_approx, f)

    return errors_dec, errors_nodec, errors_approx, jump_sizes


def plot_errors_and_jump_sizes(
    jump_sizes: Dict[str, List[float]],
    errors: Dict[str, List[float]],
    bins: List[int],
    filename: Optional[str] = None,
    x_upper: Optional[float] = 1,
    y_upper: Optional[float] = 1e-2,
) -> Tuple[matplotlib.figure.Figure, matplotlib.pyplot.Axes]:
    """Plot errors and jump sizes.

    Args:
        jump_sizes (dict): A dictionary containing the jump sizes for each grid size.
        errors (dict): A dictionary containing the errors for each grid size.
        bins (list): A list of bins for the histogram.
        filename (str, optional): The name of the file to save the plot. If None, the plot is not saved.
        x_upper (float, optional): The upper limit of the x-axis. Default is 1.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot]: A tuple containing the figure and axes objects of the plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    colors = ["black", "dimgrey", "darkgrey", "lightgrey", "gainsboro"]
    labels = [
        r"$10^3$ bins",
        r"$10^4$ bins",
        r"$10^5$ bins",
        r"$10^6$ bins",
    ]

    for i, (bin1, label) in enumerate(zip(jump_sizes.keys(), labels)):
        ax.scatter(
            jump_sizes[str(bin1)],
            errors[str(bin1)],
            s=5,
            color=colors[i],
            marker="o",
            alpha=1,
        )

    ax.set_ylabel("Relative error")
    ax.set_xlabel(r"$J_k$")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1e-24, x_upper)
    ax.set_ylim(1e-12, y_upper)
    lgnd = ax.legend(labels, loc=1)
    for handle in lgnd.legend_handles:
        handle._sizes = [90]

    if filename:
        fig.savefig(filename, bbox_inches="tight")

    return fig, ax


def calculate_integral(
    int_fn: Callable,
    bound_1: Tuple[float, float],
    bound_2: Tuple[float, float],
    limit: int = 10000000,
    epsrel: float = 1.49e-08,
):
    """Calculate the integral of a function over two separate intervals.

    This function uses the scipy.integrate.quad method to calculate the integral of the provided function over two separate intervals. The results of the two integrals are then added together.

    Args:
        int_fn (Callable): The function to integrate.
        bound_1 (Tuple[float, float]): The lower and upper bounds of the first interval.
        bound_2 (Tuple[float, float]): The lower and upper bounds of the second interval.
        limit (int, optional): An upper bound on the number of subintervals used in the computation. Default is 10000000.
        epsrel (float, optional): The desired relative error in the result. Default is 1.49e-03.

    Returns:
        float: The calculated integral of the function over the two intervals.
    """

    return (
        integrate.quad(int_fn, bound_1[0], bound_1[1], limit=limit, epsrel=epsrel)[  # type: ignore
            0
        ]
        + integrate.quad(
            int_fn, bound_2[0], bound_2[1], limit=limit, epsrel=epsrel  # type: ignore
        )[0]
    )


def _access_nested_dict(d, keys):
    for key in keys:
        d = d[str(key)]
    return d


def _initialize_nested_dict(keys):
    if not keys:
        return []
    else:
        return {str(key): _initialize_nested_dict(keys[1:]) for key in keys[0]}


def process_error_rate_vs_params(
    num_edges: List,
    params: dict,
    p_x: Callable,
    g_x: Callable,
    thresholds: List = [0.5],
    use_cache: bool = True,
    bounds=(1e-10, 1),
):
    if use_cache:
        hash_hex = hash_args(num_edges, params, p_x.__name__, g_x.__name__, thresholds)
        try:
            with open(
                f"./data/cache/{inspect.currentframe().f_code.co_name}_{hash_hex}_1.json",
                "r",
            ) as f:
                trapezium_poi_er = json.load(f)
            with open(
                f"./data/cache/{inspect.currentframe().f_code.co_name}_{hash_hex}_2.json",
                "r",
            ) as f:
                mixed_poi_er = json.load(f)
            return trapezium_poi_er, mixed_poi_er
        except FileNotFoundError:
            pass

    if len(thresholds) == 1:
        iters = [num_edges] + list(params.values())
        params_idx = 1
        if "sigma" in params.keys():
            sigma_idx = list(params).index("sigma") + 1
    else:
        iters = [num_edges] + [thresholds] + list(params.values())
        params_idx = 2
        if "sigma" in params.keys():
            sigma_idx = list(params).index("sigma") + 2
    trapezium_poi_er = _initialize_nested_dict(iters[:-1])
    mixed_poi_er = _initialize_nested_dict(iters[:-1])

    def fun(combination):
        kappa = -1 if "sigma" not in params.keys() else -1 - combination[sigma_idx]
        edges = np.logspace(-10, 0, num=combination[0], endpoint=True, base=10.0)
        if len(thresholds) == 1:
            b = edges[int(combination[0] * thresholds[0])]
            thr = thresholds[0]
        else:
            b = edges[int(combination[0] * combination[1])]
            thr = combination[1]
        p_x_func = p_x(*combination[params_idx:])
        g_x_func = g_x(*combination[params_idx:])
        r1 = calculate_integral(
            lambda x: envelope(x, p_x_func, edges, thr=thr) - p_x_func(x),
            (bounds[0], 1e-5),
            (1e-5, bounds[1]),
        )
        r2 = calculate_integral(
            lambda x: envelope(x, p_x_func, edges, g_x_func, kappa, thr=thr)
            - p_x_func(x),
            (bounds[0], 1e-5),
            (1e-5, bounds[1]),
        )
        return r1, r2

    ret = Parallel(n_jobs=-1)(
        delayed(fun)(comb) for comb in tqdm(list(itertools.product(*iters)))
    )
    for i, combination in enumerate((itertools.product(*iters))):
        _access_nested_dict(trapezium_poi_er, combination[:-1]).append(ret[i][0])
        _access_nested_dict(mixed_poi_er, combination[:-1]).append(ret[i][1])

    if use_cache:
        with open(
            f"./data/cache/{inspect.currentframe().f_code.co_name}_{hash_hex}_1.json",
            "w",
        ) as f:
            json.dump(trapezium_poi_er, f)
        with open(
            f"./data/cache/{inspect.currentframe().f_code.co_name}_{hash_hex}_2.json",
            "w",
        ) as f:
            json.dump(mixed_poi_er, f)

    return trapezium_poi_er, mixed_poi_er


def plot_poi_er_vs_num_grids(num_edges, params, mixed_poi_er, filename=None):
    linestyles = ["-", "--", "-."]
    labels = [f"c={p}" for p in params["c"]]
    iters = [num_edges] + [i for i in params.values()]
    if "sigma" in params.keys():
        fig, axs = plt.subplots(
            1, len(params["sigma"]), figsize=(4 * len(params["sigma"]), 4)
        )
        for i, s in enumerate(iters[-1]):
            for j, c in enumerate(params["c"]):
                color = "black"
                ls = linestyles[j]
                label = labels[j]
                axs[i].loglog(
                    num_edges,
                    [
                        _access_nested_dict(mixed_poi_er, [n, 1, c])[i]
                        for n in num_edges
                    ],
                    color=color,
                    ls=ls,
                    label=label,
                )

            axs[i].legend(labels)
            axs[i].set_xlabel("number of grid points")
            nm = [p for p in params.keys()][-1]
            axs[i].set_title(f"{nm}=" + str(s))
            axs[i].set_ylabel(r"$\lambda$")
            axs[i].set_xlim(0, None)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        for i, c in enumerate(params["c"]):
            color = "black"
            ls = linestyles[i]
            label = labels[i]
            ax.loglog(
                num_edges,
                [_access_nested_dict(mixed_poi_er, [n, 1])[i] for n in num_edges],
                color=color,
                ls=ls,
                label=label,
            )

        ax.legend(labels)
        ax.set_xlabel("number of grid points")
        ax.set_ylabel(r"$\lambda$")
        ax.set_xlim(0, None)

    if filename:
        fig.savefig(filename, bbox_inches="tight")


def plot_beta_process(params, ranges, filename=None, is_stable=False):
    sigma = 0
    if is_stable:
        sigma = params["sigma"]
        process = stable_beta_process(**params)
        g_x = g_stable_beta_process(**params)
    else:
        process = beta_process(**params)
        g_x = g_beta_process(**params)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (lower, upper) in zip(axs, ranges):
        xs = np.linspace(lower, upper, 100000)
        ax.plot(xs, process(xs), color="black", lw=1)
        ax.plot(
            xs,
            xs ** (-1 - sigma) * g_x(lower),
            color="black",
            lw=1,
            ls="--",
        )
        ax.plot(
            xs,
            process(xs[0])
            + (process(xs[-1]) - process(xs[0])) / 100000 * np.arange(0, 100000),
            color="black",
            ls="-.",
        )
        ax.legend(["exact", r"$x^{-1} g(x_{min})$", "trapezium"])
        # ax.set_ylim(0, None)

    if filename:
        fig.savefig(filename, bbox_inches="tight")


def plot_beta_process_err_rate_vs_m(sigmas, ms, cs, trapezium_poi, mixed_poi, filename):
    colors = ["darkgrey", "darkgrey", "darkgrey", "black", "black", "black"]
    linestyles = ["-", "--", "-.", "-", "--", "-."]
    labels_trap = [f"c={c} not decomposed" for c in cs] if trapezium_poi else []
    labels_mixed = [f"c={c} decomposed" for c in cs] if mixed_poi else []
    labels = labels_trap + labels_mixed
    cs_it = cs * 2 if (trapezium_poi is not None and mixed_poi is not None) else cs
    if trapezium_poi:
        n_grids = list(trapezium_poi.keys())[0]
    else:
        n_grids = list(mixed_poi.keys())[0]
    if sigmas is not None:
        fig, axs = plt.subplots(1, len(sigmas), figsize=(4 * len(sigmas), 4))
        for i, sigma in enumerate(sigmas):
            for j, (c, color, ls, label) in enumerate(
                zip(cs * 2, colors, linestyles, labels)
            ):
                if "not decomposed" in label:
                    axs[i].plot(
                        ms,
                        [trapezium_poi[str(n_grids)][str(m)][str(c)][i] for m in ms],
                        color=color,
                        ls=ls,
                    )
                else:
                    axs[i].plot(
                        ms,
                        [mixed_poi[str(n_grids)][str(m)][str(c)][i] for m in ms],
                        color=color,
                        ls=ls,
                    )

            axs[i].legend(labels)
            axs[i].set_xlabel("M")
            axs[i].set_title(r"sigma=" + str(sigma))
            axs[i].set_ylabel(r"$\lambda$")
            axs[i].set_xlim(0, None)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        for i, (c, color, ls, label) in enumerate(
            zip(cs_it, colors, linestyles, labels)
        ):
            if "not decomposed" in label:
                ax.plot(
                    ms,
                    [trapezium_poi[str(n_grids)][str(m)][i % len(cs)] for m in ms],
                    color=color,
                    ls=ls,
                )
            else:
                ax.plot(
                    ms,
                    [mixed_poi[str(n_grids)][str(m)][i % len(cs)] for m in ms],
                    color=color,
                    ls=ls,
                )

        ax.legend(labels)
        ax.set_xlabel("M")
        ax.set_ylabel(r"$\lambda$")
        ax.set_xlim(0, None)

    if filename:
        fig.savefig(filename, bbox_inches="tight")


def plot_error_vs_threshold_c(
    beta_error_mixed_thr, num_edges, thresholds, cs, filename=None
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    colors = ["black", "dimgrey", "darkgrey", "lightgrey", "gainsboro"]
    ls = ["-", "--", "-."]
    grids = np.logspace(-10, 0, num=101, endpoint=True, base=10.0)
    xs = [grids[int(101 * thr)] for thr in thresholds]
    for k, key in enumerate(num_edges):
        for i in range(len(cs)):
            ys = np.array(
                [beta_error_mixed_thr[str(key)][str(x)][str(1)][i] for x in thresholds]
            )
            p = "{:.0e}".format(key - 1)[-1]
            ax.loglog(
                xs,
                ys,
                c=colors[k],
                ls=ls[i],
                label=rf"$10^{p}$ bins, c={cs[i]}",
            )
            ax.set_yscale("log")
            ax.legend(loc="upper left")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(r"$\lambda$")
    if filename:
        fig.savefig(filename, bbox_inches="tight")
    return fig.tight_layout(), ax


def plot_error_vs_threshold_stable_beta_c(
    beta_error_mixed_thr, num_edges, thresholds, cs, sigmas, idx, filename=None
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    colors = ["black", "dimgrey", "darkgrey", "lightgrey", "gainsboro"]
    linestyles = ["-", "--", "-."]
    grids = np.logspace(-10, 0, num=101, endpoint=True, base=10.0)
    xs = [grids[int(101 * thr)] for thr in thresholds]
    for k, key in enumerate(num_edges):
        # for i in range(3):
        for j, (c, color, ls) in enumerate(zip(cs, colors, linestyles)):
            ys = np.array(
                [
                    beta_error_mixed_thr[str(key)][str(x)][str(1)][str(c)][idx]
                    for x in thresholds
                ]
            )
            p = "{:.0e}".format(key - 1)[-1]
            ax.loglog(
                xs,
                ys,
                c=colors[k],
                ls=linestyles[j],
                label=rf"$10^{p}$ bins, c={c}, sigma={sigmas[idx]}",
            )
            ax.set_yscale("log")
            ax.legend(loc="upper left")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(r"$\lambda$")
    if filename:
        fig.savefig(filename, bbox_inches="tight")
    return fig.tight_layout(), ax


def plot_error_vs_threshold_m(
    beta_error_mixed_thr, num_edges, thresholds, ms, filename
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    grids = np.logspace(-10, 0, num=101, endpoint=True, base=10.0)
    xs = [grids[int(101 * thr)] for thr in thresholds]
    # xs = thresholds
    colors = plt.cm.tab10.colors
    ls = ["-", "--", "-.", ":"]
    for k, key in enumerate(num_edges):
        for i, m in enumerate(ms[:4]):
            ys = np.array(
                [beta_error_mixed_thr[str(key)][str(x)][str(m)][1] for x in xs]
            )
            p = "{:.0e}".format(key - 1)[-1]
            ax.loglog(xs, ys, c=colors[k], ls=ls[i], label=rf"$10^{p}$ bins, m={m}")
            # ax.set_yscale("log")
            ax.legend(loc="upper left")

    if filename:
        fig.savefig(filename, bbox_inches="tight")
    return fig.tight_layout(), ax


def plot_error_vs_threshold_s(
    beta_error_mixed_thr, num_edges, thresholds, sigmas, filename=None
):
    grids = np.logspace(-10, 0, num=101, endpoint=True, base=10.0)
    xs = [grids[int(101 * thr)] for thr in thresholds]
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    colors = ["black", "dimgrey", "darkgrey", "lightgrey", "gainsboro"]
    ls = ["-", "--", "-."]
    for k, key in enumerate(num_edges):
        for i in range(3):
            ys = np.array(
                [
                    beta_error_mixed_thr[str(key)][str(x)][str(1)][str(2)][i]
                    for x in thresholds
                ]
            )
            p = "{:.0e}".format(key - 1)[-1]
            ax.loglog(
                xs,
                ys,
                c=colors[k],
                ls=ls[i],
                label=rf"$10^{p}$ bins, sigma={sigmas[i]}",
            )
            ax.set_yscale("log")
            ax.set_xlabel(r"$x_{\text{thr}}$")
            ax.set_ylabel(r"$\lambda$")
            ax.legend(loc="upper left")

    if filename:
        fig.savefig(filename, bbox_inches="tight")
    return fig.tight_layout(), ax
