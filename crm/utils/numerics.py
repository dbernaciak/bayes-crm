"""This module contains the numerical functions used in the simulation. """

import numpy as np
import numpy.core.numeric as nx
import numba as nb
import math


@nb.njit("f8[:](f8, f8, f8, f8)", fastmath=True, cache=True)
def logspace(start, stop, num, base=10):
    delta = stop - start
    step = delta / (num - 1)
    y = np.arange(0.0, num, dtype=np.float64)
    y = y * step
    y = y + start
    return np.power(base, y)


@nb.njit("f8(f8, f8)", fastmath=True, cache=True)
def logn(base, x):
    return np.log(x) / np.log(base)


@nb.njit("f8[:](i8)", fastmath=True, parallel=True, cache=True)
def arrival_times(size):
    ret = np.empty(size, dtype=nx.float64)
    for i in nb.prange(size):
        ret[i] = np.random.exponential()
    return np.cumsum(ret)


@nb.njit("f8[:](f8[:])", cache=True, fastmath=True)
def reverse_cumsum(x):
    return np.cumsum(x[::-1])


def beta_pdf(x, a, b):
    """Compute the probability density function of the Beta distribution.

    Args:
        x (np.ndarray): The input values.
        a (float): The shape parameter a.
        b (float): The shape parameter b.

    Returns:
        np.ndarray: The probability density function.
    """
    if not isinstance(x, np.ndarray):
        return beta_pdf_scalar(float(x), float(a), float(b))
    elif x.ndim == 1:
        return beta_pdf_1d(x, float(a), float(b))
    else:
        return beta_pdf_2d(x, float(a), float(b))


@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64), fastmath=True, cache=True)
def beta_pdf_scalar(x, a, b):

    delta = 1e-16
    log_coeff = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)

    if x == 1.0:
        x -= delta
    elif x == 0.0:
        x += delta

    if x < 0.0 or x > 1.0 or math.isnan(x):
        return 0.0
    else:
        if x != 0.0 and x != 1.0:
            log_pdf = (
                log_coeff + (a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x)
            )
        pdf = math.exp(log_pdf)  # noqa: ignore
        return pdf


@nb.njit(
    nb.float64[:](nb.float64[:], nb.float64, nb.float64),
    fastmath=True,
    parallel=True,
    cache=True,
)
def beta_pdf_1d(x, a, b):
    n = x.shape[0]
    pdf = np.empty(n, dtype=np.float64)
    log_coeff = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    for i in nb.prange(n):
        xi = x[i]
        delta = 1e-16
        if xi == 1.0:
            xi -= delta
        elif xi == 0.0:
            xi += delta

        if xi <= 0.0 or xi >= 1.0 or math.isnan(xi):
            pdf[i] = 0.0
        else:
            log_pdf = (
                log_coeff + (a - 1.0) * math.log(xi) + (b - 1.0) * math.log(1.0 - xi)
            )
            pdf[i] = math.exp(log_pdf)
    return pdf


@nb.njit(
    nb.float64[:, :](nb.float64[:, :], nb.float64, nb.float64),
    fastmath=False,
    parallel=True,
    cache=True,
)
def beta_pdf_2d(x, a, b):
    m, n = x.shape
    pdf = np.empty((m, n), dtype=np.float64)
    log_coeff = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    for i in nb.prange(m):
        for j in nb.prange(n):
            xi = x[i, j]
            delta = 1e-16
            if xi == 1.0:
                xi -= delta
            elif xi == 0.0:
                xi += delta
            if xi <= 0.0 or xi >= 1.0 or math.isnan(xi):
                pdf[i, j] = 0.0
            else:
                log_pdf = (
                    log_coeff
                    + (a - 1.0) * math.log(xi)
                    + (b - 1.0) * math.log(1.0 - xi)
                )
                pdf[i, j] = math.exp(log_pdf)
    return pdf


@nb.njit(nb.float64(nb.float64, nb.float64), fastmath=True, cache=True)
def binom(n, k):
    """Compute the binomial coefficient."""
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


@nb.njit(nb.float64(nb.float64, nb.float64), fastmath=True, cache=True)
def beta_function(a, b):
    """Compute the beta function."""
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))


@nb.njit(
    nb.float64[:](nb.float64, nb.float64, nb.int64),
    fastmath=True,
    cache=True,
)
def geospace(start, step, n):
    res = np.empty(n + 1, dtype=np.float64)
    current_value = start
    for i in range(n + 1):
        res[i] = current_value
        current_value *= step
    return res
