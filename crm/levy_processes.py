"""Levy process utilities."""
from collections.abc import Callable

import numpy as np
from scipy.special import gamma


def beta_process(m: float, c: float) -> Callable:
    """
    Beta Process.

    Args:
            m (float): A mass parameter of the Beta Process.
            c (float): A concentration parameter of the Beta Process.

        Returns:
            Callable: A function representing the Beta Process.
    """
    if not c > 0:
        raise ValueError("c must be greater than 0")  # noqa: TRY003
    if not m > 0:
        raise ValueError("m must be greater than 0")  # noqa: TRY003

    def f(x: np.ndarray) -> np.ndarray:
        return m * c * x ** (-1) * (1 - x) ** (c - 1)

    f.__name__ = "beta_process"
    f.m = m
    f.c = c
    return f


def g_beta_process(m: float, c: float) -> Callable:
    """
    Beta Process g(x).

    Args:
        m (float): A mass parameter of the Beta Process.
        c (float): A concentration parameter of the Beta Process.

    Returns:
        Callable: A function representing the g(x) of the Beta Process.

    """
    if not c > 0:
        raise ValueError("c must be greater than 0")  # noqa: TRY003
    if not m > 0:
        raise ValueError("m must be greater than 0")  # noqa: TRY003

    return lambda x: m * c * (1 - x) ** (c - 1)


def stable_beta_process(m: float, c: float, sigma: float) -> Callable:
    """
    Stable Beta Process.
    Args:
        m (float): A mass parameter of the Beta Process.
        c (float): A concentration parameter of the Beta Process.
        sigma (float): A sigma parameter of the Beta Process

    Returns:
        Callable: A function representing the Stable Beta Process.
    """
    if not c > 0:
        raise ValueError("c must be greater than 0")  # noqa: TRY003
    if not m > 0:
        raise ValueError("m must be greater than 0")  # noqa: TRY003
    if not 0 < sigma < 1:
        raise ValueError("sigma must be between 0 and 1")  # noqa: TRY003

    def f(x: np.ndarray) -> np.ndarray:
        return (
            m
            * gamma(1 + c)
            / (gamma(1 - sigma) * gamma(c + sigma))
            * x ** (-1 - sigma)
            * (1 - x) ** (c + sigma - 1)
        )

    f.__name__ = "stable_beta_process"
    f.m = m
    f.c = c
    f.sigma = sigma
    return f


def g_stable_beta_process(m: float, c: float, sigma: float) -> Callable:
    """
    Stable Beta Process.

    Args:
        m (float): A mass parameter of the Beta Process.
        c (float): A concentration parameter of the Beta Process.
        sigma (float): A sigma parameter of the Beta Process

    Returns:
        Callable: A function representing the g(x) of the Stable Beta Process.
    """
    if not c > 0:
        raise ValueError("c must be greater than 0")  # noqa: TRY003
    if not m > 0:
        raise ValueError("m must be greater than 0")  # noqa: TRY003
    if not 0 < sigma < 1:
        raise ValueError("sigma must be between 0 and 1")  # noqa: TRY003
    return (
        lambda x: m
        * gamma(1 + c)
        / (gamma(1 - sigma) * gamma(c + sigma))
        * (1 - x) ** (c + sigma - 1)
    )


def gamma_process(m: float) -> Callable:
    """
    Gamma process.

    Args:
        m (float): A mass parameter of the Gamma Process.

    Returns:
        Callable: A function representing the Gamma Process.

    """
    if not m > 0:
        raise ValueError("m must be greater than 0")  # noqa: TRY003

    def f(x: np.ndarray) -> np.ndarray:
        return m * x ** (-1) * np.exp(-x)

    f.__name__ = "gamma_process"
    f.m = m
    return f


def g_gamma_process(m: float) -> Callable:
    """
    Gamma process g(x).
    Args:
        m (float): A mass parameter of the Gamma Process.

    Returns:
        Returns a function representing the g(x) of a Gamma Process.
    """
    if not m > 0:
        raise ValueError("m must be greater than 0")  # noqa: TRY003
    return lambda x: m * np.exp(-x)


def generalized_gamma_process(m: float, sigma: float, a: float) -> Callable:
    """
    Generalized Gamma process.
    Args:
        m ():
        sigma ():
        a ():

    Returns:

    """
    if not a > 0:
        raise ValueError("a must be greater than 0")  # noqa: TRY003
    if not m > 0:
        raise ValueError("m must be greater than 0")  # noqa: TRY003
    if not 0 < sigma < 1:
        raise ValueError("sigma must be between 0 and 1")  # noqa: TRY003

    def f(x: np.ndarray) -> np.ndarray:
        return (
            m
            * (a ** (1 - sigma) / gamma(1 - sigma))
            * x ** (-1 - sigma)
            * np.exp(-a * x)
        )

    f.__name__ = "generalized_gamma_process"
    f.m = m
    f.sigma = sigma
    return f


def g_generalized_gamma_process(m: float, sigma: float, a: float) -> Callable:
    """
    Generalized Gamma process g(x).
    Args:
        m (float): A mass parameter of the Generalized Gamma Process.
        sigma (float): A sigma parameter of the Generalized Gamma Process.
        a (float): An 'a' parameter of the Generalized Gamma Process.

    Returns:

    """
    if not a > 0:
        raise ValueError("a must be greater than 0")  # noqa: TRY003
    if not m > 0:
        raise ValueError("m must be greater than 0")  # noqa: TRY003
    if not 0 < sigma < 1:
        raise ValueError("sigma must be between 0 and 1")  # noqa: TRY003
    return lambda x: m * (a ** (1 - sigma) / gamma(1 - sigma)) * np.exp(-a * x)


def sigma_stable_process(m: float, sigma: float) -> Callable:
    """
    Sigma Stable process.
    Args:
        m ():
        sigma ():

    Returns:

    """
    if not m > 0:
        raise ValueError("m must be greater than 0")  # noqa: TRY003
    if not 0 < sigma < 1:
        raise ValueError("sigma must be between 0 and 1")  # noqa: TRY003

    def f(x: np.ndarray) -> np.ndarray:
        return m * sigma / gamma(1 - sigma) * x ** (-1 - sigma)

    f.__name__ = "sigma_stable_process"
    f.m = m
    f.sigma = sigma
    return f


def g_sigma_stable_process(m: float, sigma: float) -> Callable:
    """
    Sigma Stable process g(x).
    Args:
        m ():
        sigma():

    Returns:

    """
    if not m > 0:
        raise ValueError("m must be greater than 0")  # noqa: TRY003
    if not 0 < sigma < 1:
        raise ValueError("sigma must be between 0 and 1")  # noqa: TRY003
    return lambda x: (sigma / gamma(1 - sigma)) * x ** (-1 - sigma)
