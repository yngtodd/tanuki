from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_almost_equal

import numpy as np
import pytest

from tanuki.bandit import GaussianBandit

@pytest.mark.fast_test
def test_gauss_bandit():
  """
  Quick test to build with Circle CI.
  """
  bandit = GaussianBandit()
  mu_default = 0
  sigma_default = 1

  assert_equal(bandit.mu, mu_default)
  assert_equal(bandit.sigma, sigma_default)
