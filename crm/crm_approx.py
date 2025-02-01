"""Approximation of a Levy process with a piecewise constant intensity."""

from collections.abc import Callable
from typing import Optional

import numpy as np
from math import log, log10, ceil
from scipy.integrate import quad
from crm.utils.numerics import logspace, logn, reverse_cumsum, geospace
from line_profiler_pycharm import profile


def _stable_integral(fun_eval_1, grid, kappa, idx):
    if kappa == -1:
        p_1 = fun_eval_1[:-1] * (log(grid[1] / grid[0]))
    else:
        grid_exp = grid[: idx + 1] ** (kappa + 1)
        p_1 = (
            fun_eval_1[:-1] * 1 / (kappa + 1) * (grid_exp[1 : idx + 1] - grid_exp[:idx])
        )
    return p_1


def envelope(
    x: float,
    p_x: Callable,
    edges: np.ndarray,
    g_x: Callable = None,  # noqa: RUF013
    kappa: Optional[float] = None,  # noqa: UP007
    thr=0.8,
):
    """Multipart enveloping Levy intensity.

    Args:
        x (float): Jump location.
        p_x (Callable): A function representing the process.
        edges (np.ndarray): The edges of the envelope.
        g_x (Callable): A function representing the g(x) of the process.
        kappa (float): Exponential term of the mixed approximation.
        thr (float): A threshold (between 0 and 1).

    Returns:
        float: The result of the multipart enveloping Levy intensity.
    """
    if not 0 < x <= 1:
        raise ValueError("x must be between 0 and 1")  # noqa: TRY003

    idx = int(len(edges) * thr)
    cutoff = edges[idx]
    if x in edges:
        return p_x(x)

    arg = np.argmax(edges > x)
    if g_x is not None and x < cutoff:
        before = g_x(edges[arg - 1])
        return x**kappa * before

    before = p_x(edges[arg - 1])
    if x < cutoff:
        if not kappa:
            a = p_x(edges[0])
            b = p_x(edges[1])
            c = p_x(edges[2])
            k = ((b + a) / (b + c)) * (edges[0] / edges[1])
            kappa = np.round(log10(k) / log10(edges[0] / edges[1]) - 1, 5)
        return x**kappa * before / edges[arg - 1] ** kappa

    after = p_x(edges[arg])
    return after + (before - after) * (edges[arg] - x) / (edges[arg] - edges[arg - 1])


class ApproxProcess:
    """Class to draw from a random process"""

    EPSILON = 1e-5
    CDF_EPSILON = 1e-10
    PROCESS_MIN = 1e-30
    X_LOWER = 1e-10

    def __init__(
        self,
        p_x: Callable,
        n_grids: int = 1001,
        g_x: Callable = None,  # noqa: RUF013
        kappa: float | None = None,
        edges: np.ndarray = None,
        thr: float = 0.8,
        bounds: tuple[float, float] = (0, 1),
        step: Optional[float] = None,
        do_rejection: bool = False,
    ):
        """Constructor for the ApproxProcess class.

        Args:
            p_x (Callable): Levy intensity function.
            n_grids (int): Number of grid points.
            g_x (Callable): g(x) function from Levy intensity decomposition.
            kappa (Optional[float]): Exponential term of h(x) from the mixed approximation.
            edges (np.ndarray): Grid point array.
            thr (float): Threshold for the mixed approximation.
            bounds (tuple[float, float]): Domain of the process. Default is (0, 1).
            step (Optional[float]): Step size for the grid. Default is None.
            do_rejection (bool): Whether to use rejection sampling. Default is False.
        """
        self.p_x = p_x
        self.n_grids = n_grids
        self.g_x = g_x
        self.kappa = kappa
        self.edges = edges
        self.c_sum = None
        self.thr = thr
        self.do_rejection = do_rejection
        if bounds[0] == 0:
            self.x_thr = 10 ** (-10 * (1 - 0.8))
            bounds = (ApproxProcess.X_LOWER, bounds[1])
        self.bounds = bounds
        self.logbounds = np.log10(np.array(bounds))
        if not step:
            self.step = 10 ** (log10(1 / 1e-10) / (n_grids - 1))
        else:
            self.step = step
        self.g_x_vals = None
        self.f_x_vals = None

    def _get_edges(self):
        if self.bounds[1] < np.inf:
            if self.bounds[1] != np.inf:
                self.edges = logspace(
                    self.logbounds[0],
                    self.logbounds[1],
                    num=self.n_grids,
                    # endpoint=True,
                    base=10.0,
                )
        else:
            self._extrapolate_tails()

    def _check_tail(self, x):
        gap_x = self.p_x(x)
        delta = x * (self.step - 1)
        pdf_exp = 0.5 * (self.p_x(x + delta) + gap_x) * delta
        ratio_exp = pdf_exp / (0.5 * (gap_x + self.p_x(x - delta)) * delta)
        ratio_poly = pdf_exp / (
            0.5 * (gap_x + self.p_x(x / self.step)) * x * (self.step - 1)
        )
        p = -logn(self.step, ratio_poly) - 1
        const = gap_x / x ** (-p)
        integ = quad(self.p_x, x, self.bounds[1])[0]  # noqa: ignore
        approx_integ_exp = pdf_exp / (1 - ratio_exp) - pdf_exp
        approx_integ_poly = const * x ** (-p) / p
        if abs(approx_integ_poly - integ) < ApproxProcess.EPSILON and abs(
            approx_integ_poly - integ
        ) < abs(approx_integ_exp - integ):
            extra_steps = int(logn(self.step, x / self.bounds[0]) - self.n_grids + 1)
            extra_steps += ceil(
                logn(
                    self.step, ((ApproxProcess.CDF_EPSILON * p) / const) ** (-1 / p) / x
                )
            )
            log_x = log10(
                self.bounds[0] * self.step ** (self.n_grids + extra_steps - 1)
            )
            self.edges = logspace(
                self.logbounds[0],
                log_x,
                num=self.n_grids + extra_steps,
                base=10.0,
            )
            return True

        if abs(approx_integ_exp - integ) < ApproxProcess.EPSILON:
            extra_steps = int(logn(self.step, x / self.bounds[0]) - self.n_grids + 1)
            x = self.bounds[0] * self.step ** (self.n_grids + extra_steps - 1)
            end_point = (
                ceil(
                    logn(
                        ratio_exp, ApproxProcess.CDF_EPSILON * (1 - ratio_exp) / pdf_exp
                    )
                    * delta
                    + x
                )
                * delta
                + x
            )
            self.edges = np.concatenate(
                (
                    logspace(
                        self.logbounds[0],
                        log10(x),
                        num=self.n_grids + extra_steps,
                        # endpoint=True,
                        base=10.0,
                    ),
                    np.arange(x + delta, end_point + delta, delta),
                )
            )
            return True

        return False

    def _extrapolate_tails(self):
        for x in np.logspace(
            0, 2, 5
        ):  # logspace(np.float32(0), np.float32(2.0), np.int8(5), 10):
            if self._check_tail(x):
                return

        raise ValueError("Could not extrapolate the tails")  # noqa: TRY003

    def _get_csum(self):
        if self.edges is None:
            self._get_edges()

        if self.kappa and self.g_x:
            idx = int(self.n_grids * self.thr)
            fun_eval_1 = self.g_x(self.edges[: idx + 1])
            fun_eval_2 = self.p_x(self.edges[idx:])
            p_2 = (
                (self.edges[1 + idx :] - self.edges[idx:-1])
                * (fun_eval_2[1:] + fun_eval_2[:-1])
                / 2
            )
            p_1 = _stable_integral(fun_eval_1, self.edges, self.kappa, idx)

            self.c_sum = reverse_cumsum(np.concatenate((p_1, p_2)))
            if self.do_rejection:
                self.g_x_vals = fun_eval_1
                self.f_x_vals = fun_eval_2
        elif self.find_kappa():
            idx = int(self.n_grids * self.thr)
            fun_eval = self.p_x(self.edges)
            const_1 = fun_eval[: idx + 1] / (self.edges[: idx + 1] ** self.kappa)
            p_2 = (
                (self.edges[1 + idx :] - self.edges[idx:-1])
                * (fun_eval[idx + 1 :] + fun_eval[idx:-1])
                / 2
            )
            p_1 = _stable_integral(const_1, self.edges, self.kappa, idx)
            self.c_sum = reverse_cumsum(np.concatenate((p_1, p_2)))
            if self.do_rejection:
                self.g_x_vals = const_1
                self.f_x_vals = fun_eval[idx:]
        else:
            fun_eval = self.p_x(self.edges)
            self.c_sum = reverse_cumsum(
                (self.edges[1:] - self.edges[:-1]) * (fun_eval[1:] + fun_eval[:-1]) / 2
            )
            if self.do_rejection:
                self.f_x_vals = fun_eval

    def _get_grid_extrapolated_left(self, bin_pdf, kappa, max_arrival_time):
        if round(kappa, 5) == -1:
            n = int(np.ceil((max_arrival_time - self.c_sum[-1]) / bin_pdf))
        else:
            n = int(
                np.ceil(
                    logn(
                        1 / (self.step ** (1 + kappa)),
                        1
                        - (max_arrival_time / bin_pdf)
                        * (1 - 1 / (self.step ** (1 + kappa))),
                    )
                )
            )
        min_new_grid = self.edges[0] * (1 / self.step) ** n
        if min_new_grid < self.PROCESS_MIN:
            return np.empty(0)

        return geospace(self.edges[0], 1 / self.step, n)[::-1]

    def find_kappa(self):
        if self.kappa:
            return True

        vals = self.p_x(
            np.asarray(
                [
                    1e-20,
                    1e-20 * self.step,
                    1e-20 * self.step**2,
                    1e-30,
                    1e-30 * self.step,
                    1e-30 * self.step**2,
                ]
            )
        )
        log_step_inv = log10(1 / self.step)
        k_1 = (vals[:2].sum() / vals[1:3].sum()) / self.step
        kappa_1 = log10(k_1) / log_step_inv - 1
        k_2 = (vals[3:5].sum() / vals[4:].sum()) / self.step
        kappa_2 = log10(k_2) / log_step_inv - 1
        if abs(kappa_1 - kappa_2) < 1e-5:
            self.kappa = kappa_1
            return True
        return False

    def _extend_csum(self, max_arrival_time):
        if self.c_sum[-1] < max_arrival_time:
            bin_pdf = self.c_sum[-1] - self.c_sum[-2]
            if self.kappa and self.g_x:
                extrapolated_grid = self._get_grid_extrapolated_left(
                    bin_pdf, self.kappa, max_arrival_time
                )
                if len(extrapolated_grid) > 0:
                    fun_eval_1 = self.g_x(extrapolated_grid)
                    p_1 = _stable_integral(
                        fun_eval_1,
                        extrapolated_grid,
                        self.kappa,
                        len(extrapolated_grid) - 1,
                    )
                    new_csum = reverse_cumsum(p_1) + self.c_sum[-1]
                    if self.do_rejection:
                        self.g_x_vals = np.concatenate((fun_eval_1[:-1], self.g_x_vals))
            elif self.find_kappa():
                extrapolated_grid = self._get_grid_extrapolated_left(
                    bin_pdf, self.kappa, max_arrival_time
                )
                if len(extrapolated_grid) > 0:
                    const_1 = self.p_x(extrapolated_grid) / (
                        extrapolated_grid**self.kappa
                    )
                    p_1 = _stable_integral(
                        const_1,
                        extrapolated_grid,
                        self.kappa,
                        len(extrapolated_grid) - 1,
                    )
                    new_csum = reverse_cumsum(p_1) + self.c_sum[-1]
                    if self.do_rejection:
                        self.g_x_vals = np.concatenate((const_1[:-1], self.g_x_vals))
            else:
                # This is the case where the kappa is not found, and we have to use trapezoidal rule
                n = int(np.ceil((max_arrival_time - self.c_sum[-1]) / bin_pdf))
                min_new_grid = self.edges[0] * (1 / self.step) ** n
                if min_new_grid < self.PROCESS_MIN:
                    extrapolated_grid = np.empty(0)
                else:
                    extrapolated_grid = geospace(self.edges[0], 1 / self.step, n)[::-1]
                if len(extrapolated_grid) > 0:
                    vals = self.p_x(extrapolated_grid)
                    quadrature = (
                        (vals[:-1] + vals[1:])
                        / 2
                        * (extrapolated_grid[1:] - extrapolated_grid[:-1])
                    )
                    new_csum = reverse_cumsum(quadrature) + self.c_sum[-1]
                    if self.do_rejection:
                        self.f_x_vals = np.concatenate((vals[:-1], self.f_x_vals))

            self.edges = np.concatenate((extrapolated_grid[:-1], self.edges))
            self.c_sum = np.concatenate((self.c_sum, new_csum))
            self.bounds = (self.edges[0], self.edges[-1])
            self.logbounds = np.log(np.array(self.bounds))

    @classmethod
    def from_grid(cls, intensity: np.ndarray, grid: np.ndarray):
        """Constructor from grid and corresponding intensity.

        Args:
            intensity (np.ndarray): Intensity.
            grid (np.ndarray): Grid.

        Returns:

        """
        c_sum = ((grid[1:] - grid[:-1]) * (intensity[1:] + intensity[:-1]) / 2)[
            ::-1
        ].cumsum()
        instance = cls(None, n_grids=len(grid), edges=grid)  # noqa: ignore
        instance.c_sum = c_sum
        return instance

    def _rejection_sampling_general(self, jump_sizes):
        approx_intensity = np.interp(jump_sizes, self.edges, self.f_x_vals)
        bernoulli = np.random.binomial(1, approx_intensity / self.p_x(jump_sizes))
        return jump_sizes[bernoulli == 1]

    def _rejection_sampling_stable(self, jump_sizes):
        idx = int(self.n_grids * self.thr)
        x_1 = jump_sizes[jump_sizes < self.x_thr]
        x_grid = np.asarray([self.edges[np.argmax(self.edges > j) - 1] for j in x_1])
        if self.g_x:
            approx_intensity = self.g_x(x_grid) * x_1**self.kappa
        else:
            approx_intensity = self.p_x(x_grid) * (x_1 / x_grid) ** self.kappa

        x_2 = jump_sizes[jump_sizes >= self.x_thr]
        approx_intensity = np.concatenate(
            (
                np.interp(x_2, self.edges[idx:], self.f_x_vals),
                approx_intensity,
            )
        )
        bernoulli = np.random.binomial(1, self.p_x(jump_sizes) / approx_intensity)
        return jump_sizes[bernoulli == 1]

    def _rejection_sampling(self, jump_sizes):
        if self.kappa:
            return self._rejection_sampling_stable(jump_sizes)
        else:
            return self._rejection_sampling_general(jump_sizes)

    def generate(self, arrival_times: np.ndarray = None, size: int = 100) -> np.ndarray:
        """
        Generate draws from the random process.
        Args:
            arrival_times (np.ndarray): arrival times of Poisson point process with lambda=1
            size (int): number of arrival times

        Returns: np.ndarray

        """
        if arrival_times is None:
            arrival_times = np.random.exponential(size=size).cumsum()

        if self.edges is None:
            self._get_edges()

        if self.c_sum is None:
            self._get_csum()

        self._extend_csum(arrival_times.max())
        jump_sizes = np.interp(arrival_times, self.c_sum, self.edges[:-1][::-1])
        if not self.do_rejection:
            return jump_sizes
        else:
            return self._rejection_sampling(jump_sizes)
