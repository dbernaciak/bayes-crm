"""This module contains the rejection method for the Levy processes."""
from math import gamma

import numba as nb
import numpy as np


@nb.njit("float64[:](int64, float64, float64)", fastmath=True)
def rejection_beta_brod(n, M, c):
    """Implements the rejection method for a Beta process.

    Args:
        n (int): The number of random variates to generate.
        M (float): The mass of the beta process.
        c (float): The scale of the Beta proces.

    Raises:
        Exception: If the acceptance probability is greater than 1.

    Returns:
        np.ndarray: The array contains the generated variates.
    """

    y = np.zeros(n)

    u = 0

    count = 0
    reject = 0
    while count < n:
        u = u - np.log(np.random.rand()) / M
        y_star = np.exp(-1 / c * u)
        accept = (1 - y_star) ** (c - 1)

        if accept > 1:
            raise ValueError("Acceptance probability is greater than 1")  # noqa: TRY003

        if np.random.rand() < accept:
            y[count] = y_star
            count = count + 1
        else:
            reject = reject + 1

    return y


@nb.njit("float64[:](int64, float64)", fastmath=True, cache=True)
def rejection_gamma_ros(n, M):
    """Implements the rejection method for a Gamma process.

    Args:
        n (int): The number of random variates to generate.
        M (float): The mass of the Gamma process.

    Returns:
        np.ndarray: The array contains the generated variates.
    """

    y = np.zeros(n)

    u = 0

    count = 0
    reject = 0
    while count < n:
        u = u - np.log(np.random.rand()) / M
        y_star = 1 / (np.exp(u) - 1)
        true = np.exp(-y_star)
        env = 1 / (1 + y_star)
        accept = true / env

        if np.random.rand() < accept:
            y[count] = y_star
            count = count + 1
        else:
            reject = reject + 1

    return y


@nb.njit("float64[:](int64, float64, float64)", fastmath=True, cache=True)
def rejection_gen_gamma_brod(n, M, sigma):
    """Implements the rejection method for a Generalized Gamma process.

    Args:
        n (int): The number of random variates to generate.
        M (float): The mass of the Generalized Gamma process.
        sigma (float): The sigma parameter of the Generalized Gamma process.

    Raises:
        Exception: If the acceptance probability is greater than 1.

    Returns:
        np.ndarray: The array contains the generated variates.
    """

    const1 = gamma(1 - sigma)

    y = np.zeros(n)

    u = 0

    count = 0
    reject = 0
    while count < n:
        u = u - np.log(np.random.rand()) / M
        y_star = (sigma * const1 * u) ** (-1 / sigma)
        accept = np.exp(-y_star)

        if np.random.rand() < accept:
            y[count] = y_star
            count = count + 1
        else:
            reject = reject + 1

    return y


@nb.njit("float64[:](int64, float64, float64, float64)", fastmath=True, cache=True)
def rejection_stable_beta_brod(n, M, c, sigma):
    """Implements the rejection method for a Generalized Beta process.

    Args:
        n (int): The number of random variates to generate.
        M (float): The mass of the Stable Beta process.
        c (float): The scale of the Stable Beta process.
        sigma (float): The sigma parameter of the Stable Beta process.

    Raises:
        Exception: If the acceptance probability is greater than 1.

    Returns:
        np.ndarray: The array contains the generated variates.
    """

    const1 = gamma(1 + c) / (gamma(1 - sigma) * gamma(c + sigma))

    y = np.zeros(n)

    u = 0

    count = 0
    reject = 0
    while count < n:
        u = u - np.log(np.random.rand()) / M
        y_star = (1 + u / const1 * sigma) ** (-1 / sigma)
        accept = (1 - y_star) ** (c + sigma - 1)

        if accept > 1:
            raise ValueError("Acceptance probability is greater than 1")  # noqa: TRY003

        if np.random.rand() < accept:
            y[count] = y_star
            count = count + 1
        else:
            reject = reject + 1

    return y
