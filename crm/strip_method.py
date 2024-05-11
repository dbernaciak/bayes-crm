"""Strip method for sampling of random variables."""
from typing import Callable, Tuple
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator


class StripMethod:
    """Strip method."""

    epsilon: float = 0.0
    n_grids: int = 1001

    def __init__(
        self,
        p_x: Callable = None,
        edges: np.ndarray = None,
        pdf: np.ndarray = None,
        bounds: Tuple[float, float] = (0.0, 1.0),
    ):
        self.p_x = p_x
        self.edges = edges
        if edges is not None:
            self.n_grids = len(edges)
        else:
            self.edges = np.linspace(bounds[0], bounds[1], self.n_grids, endpoint=True)
        if p_x is not None:
            fun_eval = p_x(edges)
            norm = quad(
                lambda x: p_x(x), bounds[0] + self.epsilon, bounds[1] - self.epsilon
            )[0]
            self.pdf = fun_eval / norm
        elif pdf is not None:
            self.pdf = pdf
            if p_x is None:
                self.p_x = RegularGridInterpolator(
                    (self.edges,),
                    self.pdf,
                    bounds_error=False,
                    fill_value=None,
                    method="linear",
                )
        elif pdf is None and p_x is None:
            raise AssertionError("p_x and pdf cannot be None at the same time")

        d_x = self.edges[1:] - self.edges[:-1]
        self.mins = np.amin(np.asarray([self.pdf[1:], self.pdf[:-1]]), axis=0)
        self.maxs = np.amax(np.asarray([self.pdf[1:], self.pdf[:-1]]), axis=0)
        self.probs = np.hstack((self.mins * d_x, (self.maxs - self.mins) * d_x))

    @classmethod
    def from_grid(
        cls, pdf: np.ndarray, grid: np.ndarray, bounds: Tuple[float, float] = (0, 1)
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

    @staticmethod
    def _strip(
        number_draws_in_bin,
        edges,
        rng: np.random.Generator = np.random.default_rng(seed=0),
    ) -> np.ndarray:
        """Strip method helper.

        Args:
            number_draws_in_bin (`np.ndarray`): Number of draws in each bin.
            edges (`np.ndarray`): Edges.
            rng (`np.random.Generator`): Random number generator.

        Returns:

        """
        mean_down = np.repeat(edges[:-1], number_draws_in_bin)
        mean_up = np.repeat(edges[1:], number_draws_in_bin)
        out = (
            rng.uniform(0, 1, number_draws_in_bin.sum()) * (mean_up - mean_down)
            + mean_down
        )
        return out

    def generate(
        self, size: int, rng: np.random.Generator = np.random.default_rng(seed=0)
    ) -> np.ndarray:
        """Draw random variable using the strip method.

        Args:
            size (int): Size of the sample.
            rng (`np.random.Generator`): Random number generator.

        Returns: np.ndarray

        """
        # bottom

        number_draws_in_bin = rng.multinomial(size, self.probs / self.probs.sum())
        out = self._strip(number_draws_in_bin[: self.n_grids - 1], self.edges, rng)
        # top
        mx_top = max(number_draws_in_bin[self.n_grids - 1 :])
        if mx_top > 1:
            const = 5
            out_top = self._strip(
                number_draws_in_bin[self.n_grids - 1 :] * const, self.edges, rng
            )
            idx = np.repeat(
                np.arange(0, self.n_grids - 1),
                number_draws_in_bin[self.n_grids - 1 :] * const,
            )
            unifs = rng.uniform(
                0, 1, size=(number_draws_in_bin[self.n_grids - 1 :]).sum() * const
            )
            mask_top = self.mins[idx] + (
                self.maxs[idx] - self.mins[idx]
            ) * unifs <= self.p_x(out_top)
            out_top = out_top[mask_top]
            out_top = rng.choice(out_top, number_draws_in_bin[self.n_grids - 1 :].sum())
            out = np.hstack((out, out_top))

        return out
