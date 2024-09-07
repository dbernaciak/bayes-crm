import numpy as np
import crm.two_piece_envelope as tpe


def test_two_envelope_gamma():
    np.random.seed(0)
    n = 10
    res = tpe.two_envelope_gamma(n, 2)
    assert len(res) == n
    assert np.all(res >= 0)


def test_two_envelope_gamma_mass():
    np.random.seed(0)
    n = 100
    num_iter = 1000
    res = 0
    mass = 2
    for _ in range(num_iter):
        res += tpe.two_envelope_gamma(n, mass).sum()

    assert np.round(res / num_iter,1) - 2 < 0.1


def test_two_envelope_beta():
    np.random.seed(0)
    n = 10
    res = tpe.two_envelope_beta(n, 2, 2)
    assert len(res) == n
    assert np.all(res >= 0)


def test_two_envelope_beta_mass():
    np.random.seed(0)
    n = 100
    num_iter = 1000
    res = 0
    mass = 2
    c = 2
    for _ in range(num_iter):
        res += tpe.two_envelope_beta(n, mass, c).sum()

    assert np.round(res / num_iter,1) - 2 < 0.1


def test_two_envelope_gen_gamma():
    np.random.seed(0)
    n = 10
    res = tpe.two_envelope_gen_gamma(n, 2, 0.3)
    assert len(res) == n
    assert np.all(res >= 0)


def test_two_envelope_gen_gamma_mass():
    np.random.seed(0)
    n = 100
    num_iter = 1000
    res = 0
    mass = 2
    sigma = 0.3
    for _ in range(num_iter):
        res += tpe.two_envelope_gen_gamma(n, mass, sigma).sum()

    assert np.round(res / num_iter,1) - 2 < 0.1


def test_two_envelope_stable_beta():
    np.random.seed(0)
    n = 10
    res = tpe.two_envelope_stable_beta(n, 2, 2, 0.3)
    assert len(res) == n
    assert np.all(res >= 0)


def test_two_envelope_stable_beta_mass():
    np.random.seed(0)
    n = 100
    num_iter = 1000
    res = 0
    mass = 2
    c = 2
    sigma = 0.3
    for _ in range(num_iter):
        res += tpe.two_envelope_stable_beta(n, mass, c, sigma).sum()

    assert np.round(res / num_iter,1) - 2 < 0.1