"""Strip method for sampling of random variables. Devroye (1986), page 360"""

from collections.abc import Callable
from functools import partial

import numpy as np
import numba as nb
from scipy.integrate import quad


@nb.njit("float64[:](int64[:], float64[:])", fastmath=True)
def _strip(number_draws_in_bin, edges) -> np.ndarray:
    """Strip method helper.

    Args:
        number_draws_in_bin (`np.ndarray`): Number of draws in each bin.
        edges (`np.ndarray`): Edges.

    Returns:

    """
    mean_down = np.repeat(edges[:-1], number_draws_in_bin)
    mean_up = np.repeat(edges[1:], number_draws_in_bin)
    out = (
        np.random.uniform(0, 1, number_draws_in_bin.sum()) * (mean_up - mean_down)
        + mean_down
    )
    return out


class StripMethod:
    """Strip method."""

    epsilon: float = 0.0
    n_grids: int = 1001

    def __init__(
        self,
        p_x: Callable = None,  # noqa: RUF013
        edges: np.ndarray = None,
        pdf: np.ndarray = None,
        bounds: tuple[float, float] = (0.0, 1.0),
    ):
        self.p_x = p_x
        self.edges = edges
        if edges is not None:
            self.n_grids = len(edges)
        else:
            self.edges = np.linspace(bounds[0], bounds[1], self.n_grids, endpoint=True)
        if p_x is not None and np.abs(pdf.sum() - 1) < 1e-10:
            fun_eval = p_x(self.edges)
            norm = quad(
                lambda x: p_x(x), bounds[0] + self.epsilon, bounds[1] - self.epsilon
            )[0]
            self.pdf = fun_eval / norm
        elif pdf is not None:
            self.pdf = pdf
            if p_x is None:
                self.p_x = partial(np.interp, xp=self.edges, fp=self.pdf)

        elif pdf is None and p_x is None:
            raise AssertionError(
                "p_x and pdf cannot be None at the same time"
            )  # noqa: TRY003

        d_x = self.edges[1:] - self.edges[:-1]
        self.mins = np.amin(np.asarray([self.pdf[1:], self.pdf[:-1]]), axis=0)
        self.maxs = np.amax(np.asarray([self.pdf[1:], self.pdf[:-1]]), axis=0)
        self.probs = np.hstack((self.mins * d_x, (self.maxs - self.mins) * d_x))

    @classmethod
    def from_grid(
        cls, pdf: np.ndarray, grid: np.ndarray, bounds: tuple[float, float] = (0, 1)
    ):
        """Constructor from grid and corresponding intensity.

        Args:
            pdf (np.ndarray): Probability density function.
            grid (np.ndarray): Grid.
            bounds (Tuple[float, float]): Bounds.

        Returns:

        """
        instance = cls(pdf=pdf, edges=grid, bounds=bounds)
        return instance

    def generate(
        self,
        size: int,
        rng: np.random.Generator = np.random.default_rng(seed=0),  # noqa: B008
    ) -> np.ndarray:
        """Draw random variable using the strip method.

        Args:
            size (int): Size of the sample.
            rng (`np.random.Generator`): Random number generator.

        Returns: np.ndarray

        """

        const = 10
        # bottom
        size_orig = size
        size = size * const
        number_draws_in_bin = rng.multinomial(size, self.probs / self.probs.sum())
        out = _strip(number_draws_in_bin[: self.n_grids - 1], self.edges)
        # top
        mx_top = max(number_draws_in_bin[self.n_grids - 1 :])
        if mx_top > 1:
            out_top = _strip(number_draws_in_bin[self.n_grids - 1 :], self.edges)
            idx = np.repeat(
                np.arange(0, self.n_grids - 1),
                number_draws_in_bin[self.n_grids - 1 :],
            )
            unifs = rng.uniform(
                0, 1, size=(number_draws_in_bin[self.n_grids - 1 :]).sum()
            )
            mask_top = self.mins[idx] + (
                self.maxs[idx] - self.mins[idx]
            ) * unifs <= self.p_x(out_top)
            out_top = out_top[mask_top]
            if len(out_top) > 0:
                out_top = rng.choice(
                    out_top, number_draws_in_bin[self.n_grids - 1 :].sum()
                )
                out = np.hstack((out, out_top))

        return np.random.choice(out, replace=False, size=size_orig)
