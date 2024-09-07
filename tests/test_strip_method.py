import numpy as np
import scipy.stats as st

from crm.strip_method import StripMethod


def test_monotone_dist():
    """Test the strip method for a monotone distribution."""

    a = 1
    b = 3
    N = 100_000
    np.random.seed(0)
    sm = StripMethod(p_x=st.beta(a=a, b=b).pdf, bounds=(0.0, 1.0))
    rvs = sm.generate(size=N)
    mean, var, skew, kurt = st.beta.stats(a, b, moments="mvsk")
    assert np.all(rvs >= 0.0)
    assert np.all(rvs <= 1.0)
    assert np.abs(rvs.std() - var**0.5) / var**0.5 < 1e-2
    assert np.abs((rvs.mean() - mean) / mean) < 1e-2
    assert np.abs((st.skew(rvs) - skew) / skew) < 1e-2
    assert np.abs((st.kurtosis(rvs) - kurt) / kurt) < 1e-1


def test_nonmonotone_dist():
    """Test the strip method for a non-monotone distribution."""

    a = 0
    b = 1
    N = 100_000
    np.random.seed(0)
    sm = StripMethod(p_x=st.norm(a, b).pdf, bounds=(-6, 6))
    rvs = sm.generate(size=N)
    mean, var, skew, kurt = st.norm.stats(a, b, moments="mvsk")
    assert np.all(rvs >= -6)
    assert np.all(rvs <= 6)
    assert np.abs(rvs.std() - var**0.5) / var**0.5 < 1e-2
    assert np.abs((rvs.mean() - mean)) < 1e-2
    assert np.abs((st.skew(rvs) - skew)) < 1e-2
    assert np.abs((st.kurtosis(rvs) - kurt)) < 1e-1
