"""This module contains the numerical functions used in the simulation. """
import numpy as np
import numpy.core.numeric as nx
import numba as nb


@nb.njit("f8[:](f8, f8, i8, f8)")
def logspace(start, stop, num, base=10):
    delta = stop - start
    step = delta / (num - 1)
    y = np.arange(0, num)
    y = y * step
    y = y + start
    return np.power(base, y)


@nb.njit("f8(f8, f8)")
def logn(base, x):
    return np.log(x) / np.log(base)


@nb.njit("f8[:](i8)")
def arrival_times(size):
    ret = np.empty(size, dtype=nx.float64)
    for i in range(size):
        ret[i] = np.random.exponential()
    return np.cumsum(ret)
