from sklearn.utils.testing import assert_equal
import numpy as np
import pytest


@pytest.mark.fast_test
def dummy_test():
  """
  Quick test to build with Circle CI.
  """
  x = 2 + 2
  assert_equal(x, 4)
