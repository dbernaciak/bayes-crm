"""Ferguson-Klass sampling algorithm for various processes."""
from collections.abc import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.special import exp1, gamma


# noinspection PyUnresolvedReferences
def ferguson_klass(arrival_times: np.ndarray, p_x: Callable, upper_lim: float = 1) -> np.ndarray:
    """
    Ferguson-Klass algorithm

    Args:
        arrival_times (np.ndarray): The arrival times of the unit Poisson process.
        p_x (Callable): A function representing the process.
        upper_lim (float): The upper limit of the process.

    Returns:
        np.ndarray: The result of the Ferguson-Klass algorithm.

    Returns:

    """
    jumps_exact = []
    lim = 1
    if p_x.__name__ == "gamma_process":
        return ferguson_klass_gamma(arrival_times, p_x.m)
    if p_x.__name__ == "generalized_gamma_process":
        return ferguson_klass_generalized_gamma(arrival_times, p_x.m, p_x.sigma)
    if p_x.__name__ == "beta_process":
        return ferguson_klass_beta(arrival_times, p_x.m, p_x.c)
    if p_x.__name__ == "stable_beta_process":
        return ferguson_klass_stable_beta(arrival_times, p_x.m, p_x.c, p_x.sigma)

    for t in arrival_times:

        def fun(x):
            return t - quad(p_x, x, upper_lim, limit=1000000)[0]  # noqa: B023

        try:
            jumps_exact.append(
                root_scalar(
                    fun,
                    bracket=[lim / 100, lim],
                    method="brentq",
                    x0=lim,
                    x1=1e-24,
                    xtol=1e-24,
                ).root
            )
            lim = jumps_exact[-1]
        except ValueError:
            jumps_exact.append(0)
    return np.array(jumps_exact)


def ferguson_klass_beta(arrival_times: np.ndarray, M: float, c: float) -> np.ndarray:
    """Ferguson-Klass algorithm for a Beta Process.

    Args:
        arrival_times (np.ndarray): The arrival times of the unit Poisson process.
        M (float): A mass parameter of the Beta Process.
        c (float): A concentration parameter of the Beta Process.

    Returns:
        np.ndarray: The result of the Ferguson-Klass algorithm.
    """
    u = arrival_times / M
    f = np.zeros(len(arrival_times))

    def myfun_sub(x):
        return c * np.exp((c - 1) * np.log(1 - np.exp(x)))

    def myfun(x, x0):
        integral = quad(myfun_sub, -np.exp(x), 0)[0]
        return integral - x0

    fun = lambda x: myfun(x, u[0])
    f[0] = np.exp(-np.exp(root_scalar(fun, method="brentq", bracket=[-10, 10]).root))

    for i in range(1, len(u)):
        fun = lambda x: myfun(x, u[i])  # noqa: B023
        f[i] = np.exp(
            -np.exp(
                root_scalar(
                    fun, method="brentq", bracket=[np.log(-np.log(f[i - 1])), 10]
                ).root
            )
        )

    return f


def ferguson_klass_stable_beta(arrival_times: np.ndarray, M, c, sigma):
    """Ferguson-Klass algorithm for a Stable Beta Process.

    Args:
        arrival_times (np.ndarray): The arrival times of the process.
        M (float): A mass parameter of the Stable Beta Process.
        c (float): A concentration parameter of the Stable Beta Process.
        sigma (float): A sigma parameter of the Stable Beta Process.

    Returns:
        np.ndarray: The result of the Ferguson-Klass algorithm.
    """

    const1 = gamma(1 + c) / (gamma(1 - sigma) * gamma(c + sigma))

    u = arrival_times / M
    f = np.zeros(len(arrival_times))

    def myfun_sub(x: float) -> float:
        return const1 * np.exp(-sigma * x + (c + sigma - 1) * np.log(1 - np.exp(x)))

    def myfun(x, x0):
        integral = quad(myfun_sub, -np.exp(x), 0)[0]
        return integral - x0

    fun = lambda x: myfun(x, u[0])
    f[0] = np.exp(-np.exp(root_scalar(fun, method="brentq", bracket=[-10, 10]).root))

    for i in range(1, len(u)):
        fun = lambda x: myfun(x, u[i])  # noqa: B023
        f[i] = np.exp(
            -np.exp(
                root_scalar(
                    fun, method="brentq", bracket=[np.log(-np.log(f[i - 1])), 10]
                ).root
            )
        )

    return f


def ferguson_klass_gamma(arrival_times: np.ndarray, M):
    """Ferguson-Klass algorithm for a Gamma Process.

    Args:
        arrival_times (np.ndarray): The arrival times of the process.
        M (float): A mass parameter of the Gamma Process.

    Returns:
        np.ndarray: The result of the Ferguson-Klass algorithm.
    """
    u = arrival_times / M
    f = np.zeros(len(arrival_times))

    myfun = lambda x, c: exp1(np.exp(x)) - c

    fun = lambda x: myfun(x, u[0])
    f[0] = np.exp(
        root_scalar(
            fun,
            method="secant",
            x0=0.1,
            # x1=1e-24,
            rtol=1e-15,
            maxiter=100000000,
        ).root
    )

    for i in range(1, len(u)):
        fun = lambda x: myfun(x, u[i])  # noqa: B023
        f[i] = np.exp(
            root_scalar(
                fun,
                method="secant",
                x0=np.log(f[i - 1]),
                # x1=1e-24,
                rtol=1e-15,
                maxiter=100000000,
            ).root
        )

    return f


def ferguson_klass_generalized_gamma(arrival_times, M, sigma):
    """Ferguson-Klass algorithm for a Generalized Gamma Process.

    Args:
        arrival_times (np.ndarray): The arrival times of the process.
        M (float): A mass parameter of the Generalized Gamma Process.
        sigma (float): A sigma parameter of the Generalized Gamma Process.

    Returns:
        np.ndarray: The result of the Ferguson-Klass algorithm.
    """
    u = arrival_times / M
    f = np.zeros(len(arrival_times))

    def myfun_sub(x):
        return 1 / gamma(1 - sigma) * np.exp(-sigma * x - np.exp(x))

    def myfun(x, c):
        integral = quad(myfun_sub, x, 20)[0]
        return integral - c

    fun = lambda x: myfun(x, u[0])
    f[0] = np.exp(
        root_scalar(
            fun,
            method="brentq",
            bracket=[-10, 10],
            x0=0.1,
            # x1=1e-24,
            rtol=1e-15,
            maxiter=100000,
        ).root
    )

    for i in range(1, len(u)):
        fun = lambda x: myfun(x, u[i])  # noqa: B023
        f[i] = np.exp(
            root_scalar(
                fun,
                method="secant",
                x0=np.log(f[i - 1]),
                # bracket=[np.log(f[i - 1] / 10), np.log(f[i - 1])],
                # x1=1e-24,
                rtol=1e-15,
                maxiter=100000,
            ).root
        )

    return f
