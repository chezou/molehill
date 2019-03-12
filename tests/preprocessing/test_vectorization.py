import pytest
import molehill
from molehill.preprocessing.vectorization import vectorize


@pytest.fixture()
def num_cols():
    return ['num1', 'num2']


@pytest.fixture()
def cat_cols():
    return ['cat1', 'cat2', 'cat3']


def test_vectorize(num_cols, cat_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  id
  , array_concat(
    quantitative_features(
      array("num1", "num2")
      , num1
      , num2
    ),
    categorical_features(
      array("cat1", "cat2", "cat3")
      , cat1
      , cat2
      , cat3
    )
  ) as features
  , target
from
  src_tbl
;
"""

    assert vectorize('src_tbl', 'target', cat_cols, num_cols, "id") == ret_sql


def test_vectorize_without_cols():
    with pytest.raises(ValueError):
        vectorize('src_tbl', 'target')


def test_vectorize_with_num_cols(num_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  rowid
  , quantitative_features(
    array("num1", "num2")
    , num1
    , num2
  ) as features
  , target
from
  src_tbl
;
"""

    assert vectorize('src_tbl', 'target', numerical_columns=num_cols) == ret_sql


def test_vectorize_with_cat_cols(cat_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  rowid
  , categorical_features(
    array("cat1", "cat2", "cat3")
    , cat1
    , cat2
    , cat3
  ) as features
  , target
from
  src_tbl
;
"""
    assert vectorize('src_tbl', 'target', categorical_columns=cat_cols) == ret_sql


def test_vectorize_with_bias(cat_cols, num_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  rowid
  , add_bias(
    array_concat(
      quantitative_features(
        array("num1", "num2")
        , num1
        , num2
      ),
      categorical_features(
        array("cat1", "cat2", "cat3")
        , cat1
        , cat2
        , cat3
      )
    )
  ) as features
  , target
from
  src_tbl
;
"""

    assert vectorize('src_tbl', 'target', cat_cols, num_cols, bias=True) == ret_sql


def test_vectorize_with_hashing(cat_cols, num_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  rowid
  , feature_hashing(
    array_concat(
      quantitative_features(
        array("num1", "num2")
        , num1
        , num2
      ),
      categorical_features(
        array("cat1", "cat2", "cat3")
        , cat1
        , cat2
        , cat3
      )
    )
  ) as features
  , target
from
  src_tbl
;
"""

    assert vectorize('src_tbl', 'target', cat_cols, num_cols, hashing=True) == ret_sql


def test_vectorize_with_hashing_cardinality(cat_cols, num_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  rowid
  , feature_hashing(
    array_concat(
      quantitative_features(
        array("num1", "num2")
        , num1
        , num2
      ),
      categorical_features(
        array("cat1", "cat2", "cat3")
        , cat1
        , cat2
        , cat3
      )
    )
    , '-num_features 100'
  ) as features
  , target
from
  src_tbl
;
"""

    assert vectorize('src_tbl', 'target', cat_cols, num_cols, hashing=True, feature_cardinality=100) == ret_sql


def test_vectorize_with_bias_hashing(cat_cols, num_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  rowid
  , add_bias(
    feature_hashing(
      array_concat(
        quantitative_features(
          array("num1", "num2")
          , num1
          , num2
        ),
        categorical_features(
          array("cat1", "cat2", "cat3")
          , cat1
          , cat2
          , cat3
        )
      )
    )
  ) as features
  , target
from
  src_tbl
;
"""

    assert vectorize('src_tbl', 'target', cat_cols, num_cols, bias=True, hashing=True) == ret_sql


def test_vectorize_with_emit_null(cat_cols, num_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  rowid
  , array_concat(
    quantitative_features(
      array("num1", "num2")
      , num1
      , num2
      , '-emit_null'
    ),
    categorical_features(
      array("cat1", "cat2", "cat3")
      , cat1
      , cat2
      , cat3
      , '-emit_null'
    )
  ) as features
  , target
from
  src_tbl
;
"""

    assert vectorize('src_tbl', 'target', cat_cols, num_cols, emit_null=True) == ret_sql


def test_vectorize_with_force_value(cat_cols, num_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  rowid
  , array_concat(
    quantitative_features(
      array("num1", "num2")
      , num1
      , num2
    ),
    categorical_features(
      array("cat1", "cat2", "cat3")
      , cat1
      , cat2
      , cat3
      , '-force_value'
    )
  ) as features
  , target
from
  src_tbl
;
"""

    assert vectorize('src_tbl', 'target', cat_cols, num_cols, force_value=True) == ret_sql


def test_vectorize_with_emit_null_force_value(cat_cols, num_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  rowid
  , array_concat(
    quantitative_features(
      array("num1", "num2")
      , num1
      , num2
      , '-emit_null'
    ),
    categorical_features(
      array("cat1", "cat2", "cat3")
      , cat1
      , cat2
      , cat3
      , '-emit_null -force_value'
    )
  ) as features
  , target
from
  src_tbl
;
"""

    assert vectorize('src_tbl', 'target', cat_cols, num_cols, emit_null=True, force_value=True) == ret_sql


def test_vectorize_dense(cat_cols, num_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  rowid
  , array(num1, num2, cat1, cat2, cat3) as features
  , target
from
  src_tbl
;
"""

    assert vectorize('src_tbl', 'target', cat_cols, num_cols, dense=True) == ret_sql


def test_vectorize_dense_with_hashing(cat_cols, num_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  rowid
  , array(num1, num2, mhash(cat1), mhash(cat2), mhash(cat3)) as features
  , target
from
  src_tbl
;
"""

    assert vectorize('src_tbl', 'target', cat_cols, num_cols, dense=True, hashing=True) == ret_sql


def test_vectorize_dense_with_hashing_cardinality(cat_cols, num_cols):
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  rowid
  , array(num1, num2, mhash(cat1, 100), mhash(cat2, 100), mhash(cat3, 100)) as features
  , target
from
  src_tbl
;
"""

    assert vectorize('src_tbl', 'target', cat_cols, num_cols,
                     dense=True, hashing=True, feature_cardinality=100) == ret_sql
