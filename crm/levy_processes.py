"""Levy process utilities."""
from typing import Callable
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
    assert c > 0
    assert m > 0

    def f(x):
        return m * c * x ** (-1) * (1 - x) ** (c - 1)

    f.__name__ = f"beta_process"
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
    assert c > 0
    assert m > 0
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
    assert c > 0
    assert m > 0
    assert 0 < sigma < 1

    def f(x):
        return (
            m
            * gamma(1 + c)
            / (gamma(1 - sigma) * gamma(c + sigma))
            * x ** (-1 - sigma)
            * (1 - x) ** (c + sigma - 1)
        )

    f.__name__ = f"stable_beta_process"
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
    assert c > 0
    assert m > 0
    assert 0 < sigma < 1
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
    assert m > 0

    def f(x):
        return m * x ** (-1) * np.exp(-x)

    f.__name__ = f"gamma_process"
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
    assert m > 0
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
    assert a > 0
    assert m > 0
    assert 0 < sigma < 1

    def f(x):
        return (
            m
            * (a ** (1 - sigma) / gamma(1 - sigma))
            * x ** (-1 - sigma)
            * np.exp(-a * x)
        )

    f.__name__ = f"generalized_gamma_process"
    f.m = m
    f.sigma = sigma
    return f


def g_generalized_gamma_process(m: float, sigma: float, a: float) -> Callable:
    """
    Generalized Gamma process g(x).
    Args:
        m ():
        sigma ():
        a ():

    Returns:

    """
    assert a > 0
    assert m > 0
    assert 0 < sigma < 1
    return lambda x: m * (a ** (1 - sigma) / gamma(1 - sigma)) * np.exp(-a * x)


def sigma_stable_process(m: float, sigma: float) -> Callable:
    """
    Sigma Stable process.
    Args:
        m ():
        sigma ():

    Returns:

    """
    assert m > 0
    assert 0 < sigma < 1

    def f(x):
        return m * sigma / gamma(1 - sigma) * x ** (-1 - sigma)

    f.__name__ = f"sigma_stable_process"
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
    assert m > 0
    assert 0 < sigma < 1
    return lambda x: (sigma / gamma(1 - sigma)) * x ** (-1 - sigma)
