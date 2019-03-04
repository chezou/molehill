import pytest
from molehill.preprocessing.normalization import Normalizer


@pytest.fixture()
def num_cols():
    return ['num1', 'num2']


def test_normalizer_minmax(num_cols):
    ret_sql = """\
rescale(
  num1
  , ${td.last_results.num1_min_train}
  , ${td.last_results.num1_max_train}
) as num1
, rescale(
  num2
  , ${td.last_results.num2_min_train}
  , ${td.last_results.num2_max_train}
) as num2"""

    inv_sql = """\
num1 * (${td.last_results.num1_max_train} - ${td.last_results.num1_min_train})
  + ${td.last_results.num1_min_train} as num1
, num2 * (${td.last_results.num2_max_train} - ${td.last_results.num2_min_train})
  + ${td.last_results.num2_min_train} as num2"""

    normalizer = Normalizer("minmax", "train")
    assert normalizer.transform(num_cols) == ret_sql
    assert normalizer.invert_transform(num_cols) == inv_sql


def test_normalizer_minmax_without_phase(num_cols):
    ret_sql = """\
rescale(
  num1
  , ${td.last_results.num1_min}
  , ${td.last_results.num1_max}
) as num1
, rescale(
  num2
  , ${td.last_results.num2_min}
  , ${td.last_results.num2_max}
) as num2"""

    inv_sql = """\
num1 * (${td.last_results.num1_max} - ${td.last_results.num1_min})
  + ${td.last_results.num1_min} as num1
, num2 * (${td.last_results.num2_max} - ${td.last_results.num2_min})
  + ${td.last_results.num2_min} as num2"""

    normalizer = Normalizer("minmax", None)
    assert normalizer.transform(num_cols) == ret_sql
    assert normalizer.invert_transform(num_cols) == inv_sql


def test_normalizer_standardize(num_cols):
    ret_sql = """\
zscore(
  num1
  , ${td.last_results.num1_mean_train}
  , ${td.last_results.num1_std_train}
) as num1
, zscore(
  num2
  , ${td.last_results.num2_mean_train}
  , ${td.last_results.num2_std_train}
) as num2"""

    inv_sql = """\
num1 * ${td.last_results.num1_std_train} + ${td.last_results.num1_mean_train} as num1
, num2 * ${td.last_results.num2_std_train} + ${td.last_results.num2_mean_train} as num2"""

    normalizer = Normalizer("standardize", "train")
    assert normalizer.transform(num_cols) == ret_sql
    assert normalizer.invert_transform(num_cols) == inv_sql


def test_normalize_log1p(num_cols):
    ret_sql = """\
ln(num1 + 1) as num1
, ln(num2 + 1) as num2"""

    inv_sql = """\
exp(num1) - 1 as num1
, exp(num2) - 1 as num2"""

    normalizer = Normalizer("log1p")
    assert normalizer.transform(num_cols) == ret_sql
    assert normalizer.invert_transform(num_cols) == inv_sql
