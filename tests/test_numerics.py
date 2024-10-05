"""Tests for crm.utils.numerics module."""

import numpy as np
import numpy.core.numeric as nx
import numba as nb
import math
import scipy.stats as stats

from crm.utils.numerics import logspace, logn, arrival_times, reverse_cumsum, beta_pdf


def test_logspace():
    assert all(
        abs(logspace(0.9, 1.9, 11.0, 10.0) - np.logspace(0.9, 1.9, 11, True, base=10))
        < 1e-10
    )
    assert all(
        abs(logspace(0.0, 1.0, 11.0, 2.0) - np.logspace(0.0, 1.0, 11, True, base=2))
        < 1e-10
    )
    assert all(
        abs(logspace(0.0, 1.0, 11.0, 3.0) - np.logspace(0.0, 1.0, 11, True, base=3))
        < 1e-10
    )
    assert all(
        abs(logspace(0.0, 1.0, 11.0, 4.0) - np.logspace(0.0, 1.0, 11, True, base=4))
        < 1e-10
    )
    assert all(
        abs(logspace(0.0, 1.0, 11.0, 5.0) - np.logspace(0.0, 1.0, 11, True, base=5))
        < 1e-10
    )
    assert all(
        abs(logspace(0.0, 1.0, 11.0, 6.0) - np.logspace(0.0, 1.0, 11, True, base=6))
        < 1e-10
    )
    assert all(
        abs(logspace(0.0, 1.0, 11.0, 7.0) - np.logspace(0.0, 1.0, 11, True, base=7))
        < 1e-10
    )
    assert all(
        abs(logspace(0.0, 1.0, 11.0, 8.0) - np.logspace(0.0, 1.0, 11, True, base=8))
        < 1e-10
    )
    assert all(
        abs(logspace(0.0, 1.0, 11.0, 9.0) - np.logspace(0.0, 1.0, 11, True, base=9))
        < 1e-10
    )


def test_logn():

    assert abs(logn(2, 8) - 3) < 1e-10
    assert abs(logn(3, 27) - 3) < 1e-10
    assert abs(logn(4, 64) - 3) < 1e-10
    assert abs(logn(5, 125) - 3) < 1e-10
    assert abs(logn(6, 216) - 3) < 1e-10
    assert abs(logn(7, 343) - 3) < 1e-10
    assert abs(logn(8, 512) - 3) < 1e-10
    assert abs(logn(9, 729) - 3) < 1e-10
    assert abs(logn(10, 1000) - 3) < 1e-10

    assert abs(logn(2, 16) - 4) < 1e-10
    assert abs(logn(3, 81) - 4) < 1e-10
    assert abs(logn(4, 256) - 4) < 1e-10
    assert abs(logn(5, 625) - 4) < 1e-10
    assert abs(logn(6, 1296) - 4) < 1e-10
    assert abs(logn(7, 2401) - 4) < 1e-10
    assert abs(logn(8, 4096) - 4) < 1e-10
    assert abs(logn(9, 6561) - 4) < 1e-10
    assert abs(logn(10, 10000) - 4) < 1e-10

    assert abs(logn(2, 32) - 5) < 1e-10
    assert abs(logn(3, 243) - 5) < 1e-10
    assert abs(logn(4, 1024) - 5) < 1e-10
    assert abs(logn(5, 3125) - 5) < 1e-10
    assert abs(logn(6, 7776) - 5) < 1e-10
    assert abs(logn(7, 16807) - 5) < 1e-10
    assert abs(logn(8, 32768) - 5) < 1e-10
    assert abs(logn(9, 59049) - 5) < 1e-10
    assert abs(logn(10, 100000) - 5) < 1e-10

    assert abs(logn(2, 64) - 6) < 1e-10
    assert abs(logn(3, 729) - 6) < 1e-10
    assert abs(logn(4, 4096) - 6) < 1e-10
    assert abs(logn(5, 15625) - 6) < 1e-10
    assert abs(logn(6, 46656) - 6) < 1e-10
    assert abs(logn(7, 117649) - 6) < 1e-10
    assert abs(logn(8, 262144) - 6) < 1e-10
    assert abs(logn(9, 531441) - 6) < 1e-10
    assert abs(logn(10, 1000000) - 6) < 1e-10

    assert abs(logn(2, 128) - 7) < 1e-10
    assert abs(logn(3, 2187) - 7) < 1e-10
    assert abs(logn(4, 16384) - 7) < 1e-10
    assert abs(logn(5, 78125) - 7) < 1e-10
    assert abs(logn(6, 279936) - 7) < 1e-10
    assert abs(logn(7, 823543) - 7) < 1e-10
    assert abs(logn(8, 2097152) - 7) < 1e-10
    assert abs(logn(9, 4782969) - 7) < 1e-10
    assert abs(logn(10, 10000000) - 7) < 1e-10

    assert abs(logn(2, 256) - 8) < 1e-10
    assert abs(logn(3, 6561) - 8) < 1e-10
    assert abs(logn(4, 65536) - 8) < 1e-10
    assert abs(logn(5, 390625) - 8) < 1e-10
    assert abs(logn(6, 1679616) - 8) < 1e-10
    assert abs(logn(7, 5764801) - 8) < 1e-10
    assert abs(logn(8, 16777216) - 8) < 1e-10
    assert abs(logn(9, 43046721) - 8) < 1e-10
    assert abs(logn(10, 100000000) - 8) < 1e-10


def test_beta_pdf():
    """Test the beta_pdf function against scipy."""
    x = np.linspace(1e-10, 1.0 - 1e-10, 101)
    a = 1.0
    b = 1.0
    assert all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 2.0
    b = 2.0
    assert all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 3.0
    b = 3.0
    assert all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 4.0
    b = 4.0
    assert all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 5.0
    b = 5.0
    assert all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 6.0
    b = 6.0
    assert all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 7.0
    b = 7.0
    assert all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 8.0
    b = 8.0
    assert all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 9.0
    b = 9.0
    assert all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 10.0
    b = 10.0
    assert all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)

    # test scalar x
    x = 0.5
    a = 1.0
    b = 1.0
    assert abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10
    a = 2.0
    b = 2.0
    assert abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10
    a = 3.0
    b = 3.0
    assert abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10
    a = 4.0
    b = 4.0
    assert abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10
    a = 5.0
    b = 5.0
    assert abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10
    a = 6.0
    b = 6.0
    assert abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10
    a = 7.0
    b = 7.0
    assert abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10
    a = 8.0
    b = 8.0
    assert abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10
    a = 9.0
    b = 9.0
    assert abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10
    a = 10.0
    b = 10.0
    assert abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10

    # test 2 dim x
    x = np.array([[0.1, 0.5], [0.2, 0.6]])
    a = 1.0
    b = 1.0
    assert np.all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 2.0
    b = 2.0
    assert np.all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 3.0
    b = 3.0
    assert np.all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 4.0
    b = 4.0
    assert np.all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 5.0
    b = 5.0
    assert np.all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 6.0
    b = 6.0
    assert np.all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 7.0
    b = 7.0
    assert np.all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 8.0
    b = 8.0
    assert np.all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 9.0
    b = 9.0
    assert np.all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
    a = 10.0
    b = 10.0
    assert np.all(abs(beta_pdf(x, a, b) - stats.beta.pdf(x, a, b)) < 1e-10)
