import pytest
from molehill.preprocessing.impute import Imputer


@pytest.fixture()
def num_cols():
    return ['num1', 'num2']


@pytest.fixture()
def cat_cols():
    return ['cat1', 'cat2']


def test_numeric_imputer(num_cols):
    ret_sql = """\
coalesce(num1, ${td.last_results.num1_mean_train}) as num1
, coalesce(num2, ${td.last_results.num2_mean_train}) as num2"""

    numeric_imputer = Imputer('mean', 'train')
    assert numeric_imputer.transform(num_cols) == ret_sql


def test_categorical_imputer(cat_cols):
    ret_sql = """\
coalesce(cast(cat1 as varchar), 'missing') as cat1
, coalesce(cast(cat2 as varchar), 'missing') as cat2"""

    categorical_imputer = Imputer('constant', 'train', 'missing', categorical=True)
    assert categorical_imputer.transform(cat_cols) == ret_sql


def test_numeric_imputer_without_phase(num_cols):
    ret_sql = """\
coalesce(num1, ${td.last_results.num1_mean}) as num1
, coalesce(num2, ${td.last_results.num2_mean}) as num2"""

    numeric_imputer = Imputer('mean', None)
    assert numeric_imputer.transform(num_cols) == ret_sql
