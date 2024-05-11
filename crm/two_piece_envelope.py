import numpy as np
import numba as nb
from math import gamma


@nb.njit("float64[:](int64, float64, float64)", fastmath=True)
def two_envelope_beta(n, M, c):
    """Implements the two-envelope acceptance-rejection method for a Beta process.

    Args:
        n (int): The number of random variates to generate.
        M (float): The mass of the beta process.
        c (float): The scale of the Beta proces.

    Raises:
        Exception: If the acceptance probability is greater than 1.

    Returns:
        np.ndarray: The array contains the generated variates.
    """

    b = 0.8 / c
    const1 = b ** (-1) * (1 - b) ** c

    y = np.zeros(n)

    u = 0

    count = 0
    reject = 0
    while count < n:
        u = u - np.log(np.random.rand()) / M
        if u > const1:
            ystar = b * np.exp(-1 / c * (u - const1))
            accept = (1 - ystar) ** (c - 1)
        else:
            ystar = 1 - (u * b) ** (1 / c)
            accept = b * ystar ** (-1)

        if accept > 1:
            raise Exception("Acceptance probability is greater than 1")

        if np.random.rand() < accept:
            y[count] = ystar
            count += 1
        else:
            reject += 1

    return y


@nb.njit("float64[:](int64, float64)", fastmath=True)
def two_envelope_gamma(n, M):
    """Implements the two-envelope acceptance-rejection method for a Gamma process.

    Args:
        n (int): The number of random variates to generate.
        M (float): The mass of the Gamma process.

    Raises:
        Exception: If the acceptance probability is greater than 1.

    Returns:
        np.ndarray: The array contains the generated variates.
    """
    b = 0.8065
    logb = np.log(b)
    const1 = np.exp(-b) / b

    y = np.zeros(n)

    u = 0

    count = 0
    reject = 0
    while count < n:
        u = u - np.log(np.random.rand()) / M
        if u > const1:
            ystar = b * np.exp(const1 - u)
            accept = np.exp(-ystar)
        else:
            ystar = -logb - np.log(u)
            accept = b * ystar ** (-1)

        if np.random.rand() < accept:
            count = count + 1
            y[count - 1] = ystar
        else:
            reject = reject + 1

    return y


@nb.njit("float64[:](int64, float64, float64, float64)", fastmath=True)
def two_envelope_stable_beta(n, M, c, sigma):
    """Implements the two-envelope acceptance-rejection method for a Stable Beta process.

    Args:
        n (int): The number of random variates to generate.
        M (float): The mass of the Stable Beta process.
        c (float): The scale of the Stable Beta proces.
        sigma (float): The sigma parameter of the Stable Beta process.

    Raises:
        Exception: If the acceptance probability is greater than 1.

    Returns:
        np.ndarray: The array contains the generated variates.
    """
    b = 0.8 / c

    const1 = gamma(1 + c) / (gamma(1 - sigma) * gamma(c + sigma))
    const2 = const1 / (c + sigma)
    const3 = b ** (-sigma) - sigma / (c + sigma) * b ** (-1 - sigma) * (1 - b) ** (
        c + sigma
    )
    const4 = const2 * b ** (-1 - sigma) * (1 - b) ** (c + sigma)

    y = np.zeros(n)

    u = 0

    count = 0
    reject = 0
    while count < n:
        u = u - np.log(np.random.rand()) / M
        if u > const4:
            ystar = (u / const1 * sigma + const3) ** (-1 / sigma)
            accept = (1 - ystar) ** (c + sigma - 1)
        else:
            ystar = 1 - (u / const2 * b ** (1 + sigma)) ** (1 / (c + sigma))
            accept = (ystar / b) ** (-1 - sigma)

        if accept > 1:
            raise Exception("Acceptance probability is greater than 1")

        if np.random.rand() < accept:
            count = count + 1
            y[count - 1] = ystar
        else:
            reject = reject + 1

    return y


@nb.njit("float64[:](int64, float64, float64)", fastmath=True)
def two_envelope_gen_gamma(n, M, sigma):
    """Implements the two-envelope acceptance-rejection method for a Generalized Gamma process.

    Args:
        n (int): The number of random variates to generate.
        M (float): The mass of the Generalized Gamma process.
        sigma (float): The sigma parameter of the Generalized Gamma process.

    Raises:
        Exception: If the acceptance probability is greater than 1.

    Returns:
        np.ndarray: The array contains the generated variates.
    """

    b = 0.8065

    const1 = gamma(1 - sigma)
    const2 = b ** (-1 - sigma) * np.exp(-b) / const1
    const3 = b ** (-sigma) - sigma * b ** (-1 - sigma) * np.exp(-b)
    const4 = -(sigma + 1) * np.log(b) - const1

    y = np.zeros(n)

    u = 0

    count = 0
    reject = 0
    while count < n:
        u = u - np.log(np.random.rand()) / M
        if u > const2:
            ystar = (sigma * const1 * u + const3) ** (-1 / sigma)
            accept = np.exp(-ystar)
        else:
            ystar = const4 - np.log(u)
            accept = (ystar / b) ** (-1 - sigma)

        if np.random.rand() < accept:
            y[count] = ystar
            count = count + 1
        else:
            reject = reject + 1

    return y
