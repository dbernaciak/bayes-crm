"""Approximation of a Levy process with a piecewise constant intensity."""
from collections.abc import Callable
from typing import Optional

import numpy as np
from math import log, log10
from scipy.integrate import quad
from crm.utils.numerics import logspace, logn, reverse_cumsum


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
        # after = g_x(edges[arg])
        return x**kappa * before

    before = p_x(edges[arg - 1])
    after = p_x(edges[arg])
    return after + (before - after) * (edges[arg] - x) / (edges[arg] - edges[arg - 1])


class ApproxProcess:
    """Class to draw from a random process"""

    EPSILON = 1e-5
    CDF_EPSILON = 1e-10

    def __init__(
        self,
        p_x: Callable,
        n_grids: int = 1001,
        g_x: Callable = None,  # noqa: RUF013
        kappa: Optional[float] = None,  # noqa: UP007
        edges: np.ndarray = None,
        thr: float = 0.8,
        bounds: tuple[float, float] = (1e-10, 1),
    ):
        self.p_x = p_x
        self.n_grids = n_grids
        self.g_x = g_x
        self.kappa = kappa
        self.edges = edges
        self.c_sum = None
        self.thr = thr
        if bounds[0] == 0:
            bounds = (1e-10, bounds[1])
        self.bounds = bounds
        self.logbounds = np.log10(np.array(bounds))
        self.step = 10 ** (10 / (n_grids - 1))

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
        ratio_exp = pdf_exp / (0.5 * (self.p_x(x) + self.p_x(x - delta)) * delta)
        ratio_poly = pdf_exp / (0.5 * (gap_x + self.p_x(x / self.step)) * x * (self.step - 1))
        p = -logn(self.step, ratio_poly) - 1
        const = gap_x / x ** (-p)
        integ = quad(self.p_x, x, self.bounds[1])[0]
        approx_integ_exp = pdf_exp / (1 - ratio_exp) - pdf_exp
        approx_integ_poly = const * x ** (-p) / p
        if abs(approx_integ_poly - integ) < ApproxProcess.EPSILON and abs(
            approx_integ_poly - integ
        ) < abs(approx_integ_exp - integ):
            extra_steps = int(logn(self.step, 1e6 / self.bounds[0]) - self.n_grids + 1)  # needs optimization
            x = self.bounds[0] * self.step ** (self.n_grids + extra_steps - 1)
            self.edges = logspace(
                self.logbounds[0],
                log10(x),
                num=self.n_grids + extra_steps,
                base=10.0,
            )
            return True

        if abs(approx_integ_exp - integ) < ApproxProcess.EPSILON:
            extra_steps = int(logn(self.step, x / self.bounds[0]) - self.n_grids + 1)
            x = self.bounds[0] * self.step ** (self.n_grids + extra_steps - 1)
            exp_bins = pdf_exp * ratio_exp ** np.arange(1, 1001)
            end_point = (
                np.argwhere(
                    reverse_cumsum(exp_bins)[::-1]
                    < ApproxProcess.CDF_EPSILON / ratio_exp
                ).ravel()[0]
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

        if self.kappa is None or self.g_x is None:
            fun_eval = self.p_x(self.edges)
            self.c_sum = reverse_cumsum(
                (self.edges[1:] - self.edges[:-1]) * (fun_eval[1:] + fun_eval[:-1]) / 2
            )
        else:
            idx = int(self.n_grids * self.thr)
            fun_eval_1 = self.g_x(self.edges[: idx + 1])
            fun_eval_2 = self.p_x(self.edges[idx:])
            p_2 = (
                (self.edges[1 + idx :] - self.edges[idx:-1])
                * (fun_eval_2[1:] + fun_eval_2[:-1])
                / 2
            )

            if self.kappa == -1:
                p_1 = fun_eval_1[:-1] * (log(self.edges[1] / self.edges[0]))
            else:
                p_1 = (
                    fun_eval_1[:-1]
                    * 1
                    / (self.kappa + 1)
                    * (
                        self.edges[1 : idx + 1] ** (self.kappa + 1)
                        - self.edges[:idx] ** (self.kappa + 1)
                    )
                )

            self.c_sum = reverse_cumsum(np.concatenate((p_1, p_2)))

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
        return logspace(
            logn(self.step, min_new_grid),
            logn(self.step, self.edges[0]),
            num=n + 1,
            # endpoint=True,
            base=self.step,
        )

    def _extend_csum(self, max_arrival_time):
        if self.c_sum[-1] < max_arrival_time:
            bin_pdf = self.c_sum[-1] - self.c_sum[-2]
            if self.kappa and self.g_x:
                extrapolated_grid = self._get_grid_extrapolated_left(
                    bin_pdf, self.kappa, max_arrival_time
                )
                fun_eval_1 = self.g_x(extrapolated_grid)
                if self.kappa == -1:
                    p_1 = fun_eval_1[:-1] * (log(self.edges[1] / self.edges[0]))
                else:
                    p_1 = (
                        fun_eval_1[:-1]
                        * 1
                        / (self.kappa + 1)
                        * (
                            extrapolated_grid[1:] ** (self.kappa + 1)
                            - extrapolated_grid[:-1] ** (self.kappa + 1)
                        )
                    )
                new_csum = reverse_cumsum(p_1) + self.c_sum[-1]
            else:
                # back-out the exponential term
                k = (self.c_sum[-1] - self.c_sum[-2]) / (
                    self.c_sum[-2] - self.c_sum[-3]
                )
                self.kappa = log10(k) / log10(self.edges[0] / self.edges[1]) - 1
                extrapolated_grid = self._get_grid_extrapolated_left(
                    bin_pdf, self.kappa, max_arrival_time
                )
                if round(self.kappa, 5) != -1:
                    const_1 = self.p_x(extrapolated_grid[:-1]) / (
                        extrapolated_grid[:-1] ** (self.kappa)
                    )
                    p_1 = (
                        const_1
                        * 1
                        / (self.kappa + 1)
                        * (
                            extrapolated_grid[1:] ** (self.kappa + 1)
                            - extrapolated_grid[:-1] ** (self.kappa + 1)
                        )
                    )
                    new_csum = reverse_cumsum(p_1) + self.c_sum[-1]
                else:
                    n = len(extrapolated_grid[:-1])
                    new_csum = (np.ones(n) * bin_pdf).cumsum() + self.c_sum[-1]

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
        instance = cls(None, n_grids=len(grid), edges=grid)
        instance.c_sum = c_sum
        return instance

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

        return np.interp(arrival_times, self.c_sum, self.edges[:-1][::-1])
