"""Occupancy model."""
from collections.abc import Callable

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ..crm_approx import ApproxProcess
from ..strip_method import StripMethod


class OccupancyModel:
    """Occupancy model as presented in more for less paper."""

    n_grids_theta = 1001
    n_grids_q = 1001
    precision = 1e-10

    def __init__(self, prior_theta, prior_q, num_sites, num_occasions):
        self.prior_theta = prior_theta
        self.prior_q = prior_q
        self.num_sites = num_sites
        self.num_occasions = num_occasions
        self.theta_grid = np.logspace(
            np.log10(OccupancyModel.precision),
            0,
            num=OccupancyModel.n_grids_theta,
            endpoint=True,
            base=10.0,
        )
        self.q_grid = np.linspace(0, 1, num=OccupancyModel.n_grids_q, endpoint=True)
        self.theta_mesh, self.q_mesh = np.meshgrid(self.theta_grid, self.q_grid)

    def _posterior_discovered(self, observations, qm_3d):
        return (
            self.prior_theta(self.theta_mesh.T)
            * self.prior_q(self.q_mesh.T)
            * (
                (self.theta_mesh.T.reshape(*self.theta_mesh.T.shape, 1) * qm_3d)
                + (
                    (1 - self.theta_mesh.T.reshape(*self.theta_mesh.T.shape, 1))
                    * (1 * (observations.sum(axis=0) == 0)).reshape(1, self.num_sites)
                )
            ).prod(axis=2)
        )

    def _posterior_undiscovered(self):
        return (
            self.prior_theta(self.theta_mesh)
            * self.prior_q(self.q_mesh)
            * (
                (
                    self.theta_mesh.reshape(*self.theta_mesh.shape, 1)
                    * (1 - self.q_mesh).reshape(*self.theta_mesh.shape, 1)
                    ** self.num_occasions
                )
                + (
                        1
                        - self.theta_mesh.reshape(
                            *self.theta_mesh.shape,
                            1,
                        )
                )
            ).prod(axis=2)
        )

    def posterior(self, observations: np.ndarray) -> tuple[np.ndarray, Callable]:
        """Posterior calculator.

        Args:
            observations (class:`np.ndarray`):

        Returns: Tuple of grid of marginal theta posterior and
            callable of q conditional on theta posterior.

        """
        qm_2d = np.asarray(
            [np.where((observations == 0), 1 - q, q) for q in self.q_grid]
        )
        tmp_prd = qm_2d.prod(axis=1)
        qm_3d = np.asarray([tmp_prd for _ in range(OccupancyModel.n_grids_theta)])
        if np.any(observations):
            posterior_discovered = self._posterior_discovered(observations, qm_3d)
            marginal_theta = np.trapz(posterior_discovered, self.q_grid, axis=1)
            norm_theta = np.trapz(marginal_theta, self.theta_grid, axis=0)
            marginal_theta = marginal_theta / norm_theta

            def conditional_qq(theta_sampled):
                return self.prior_q(self.q_grid) * (
                    (theta_sampled * qm_2d.prod(axis=1))
                    + (
                        (1 - theta_sampled)
                        * (1 * (observations.sum(axis=0) == 0)).reshape(
                            1, self.num_sites
                        )
                    )
                ).prod(axis=1)

            def conditional_q(theta_sampled):
                return conditional_qq(theta_sampled) / np.trapz(
                    conditional_qq(theta_sampled), self.q_grid, axis=0
                )

        else:
            posterior_undiscovered = self._posterior_undiscovered()
            marginal_theta = np.trapz(posterior_undiscovered, self.q_mesh, axis=0)

            def conditional_qq(theta_sampled):
                return self.prior_q(self.q_grid) * (
                    (theta_sampled * qm_2d.prod(axis=1) ** self.num_occasions)
                    + (1 - theta_sampled)
                ).prod(axis=1)

            def conditional_q(theta_sampled):
                return conditional_qq(theta_sampled) / np.trapz(
                    conditional_qq(theta_sampled), self.q_grid, axis=0
                )

        return marginal_theta, conditional_q


def predictive(
    occupancy_model: OccupancyModel,
    theta: np.ndarray,
    q: np.ndarray,
    n_draws: int,
    n_sites: int,
    n_samplings: int,
    n_extra_samplings: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Predictive distribution.

    Args:
        occupancy_model (class:`OccupancyModel`): Occupancy model.
        theta (class:`np.ndarray`): Marginal theta posterior
        q (class:`np.ndarray`): Conditional q posterior
        n_draws (int): Number of draws
        n_sites (int): Number of sites
        n_samplings (int): Number of samplings
        n_extra_samplings (int): Number of extra samplings

    Returns:
        class:`Tuple[np.ndarray, np.ndarray]`:
    """

    y_obs = []
    counter = 0
    y = np.zeros((n_samplings, n_sites))
    for _, (theta_i, q_i) in tqdm(enumerate(zip(theta, q, strict=False))):
        z = np.random.binomial(n=1, p=theta_i, size=n_sites)
        y = np.random.binomial(n=1, p=z * np.ones((n_samplings, n_sites)) * q_i)
        if np.any(y):
            theta_posterior, q_cond_posterior = occupancy_model.posterior(y)
            theta_sm = StripMethod.from_grid(
                theta_posterior, occupancy_model.theta_grid
            )
            thetas_gen = theta_sm.generate(size=n_draws)
            y_obs.append(np.empty((n_draws, n_sites)))
            for j, theta_gen in enumerate(thetas_gen):
                q_pdf = q_cond_posterior(theta_gen)
                q_sm = StripMethod.from_grid(q_pdf, occupancy_model.q_grid)
                q_gen = q_sm.generate(size=10)
                np.random.shuffle(q_gen)
                q_gen = q_gen[0]
                z_gen = np.random.binomial(n=1, p=theta_gen, size=n_sites)
                y_gen = np.random.binomial(
                    n=1, p=z_gen * np.ones((n_extra_samplings, n_sites)) * q_gen
                )
                y_obs[-1][j] = y_gen.sum(axis=0)
            counter += 1

    theta_posterior, q_cond_posterior = occupancy_model.posterior(np.zeros(y.shape))
    theta_sm = ApproxProcess.from_grid(theta_posterior, occupancy_model.theta_grid)
    n_unobs = 50

    def gen_unobs(theta_sm, q_cond_posterior, n_unobs, n_sites, n_extra_samplings) -> np.ndarray:
        y_unobs = np.empty((n_unobs, n_sites))
        unobserved_theta = theta_sm.generate(size=n_unobs)
        for i, u_theta in enumerate(unobserved_theta):
            q_pdf = q_cond_posterior(u_theta)
            q_sm = StripMethod.from_grid(q_pdf, occupancy_model.q_grid)
            q_gen = q_sm.generate(size=10)
            np.random.shuffle(q_gen)
            q_gen = q_gen[0]
            z_gen = np.random.binomial(n=1, p=u_theta, size=n_sites)
            y_gen = np.random.binomial(
                n=1, p=z_gen * np.ones((n_extra_samplings, n_sites)) * q_gen
            )
            y_unobs[i] = y_gen.sum(axis=0)
        return y_unobs

    y_unobs = Parallel(n_jobs=-1)(
        delayed(gen_unobs)(
            theta_sm, q_cond_posterior, n_unobs, n_sites, n_extra_samplings
        )
        for _ in tqdm(range(n_draws))
    )
    y_unobs = np.asarray(y_unobs).transpose(1, 0, 2)

    return np.asarray(y_obs), y_unobs
