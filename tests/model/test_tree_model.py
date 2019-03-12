import pytest
import molehill
from molehill.model import train_randomforest_classifier, train_randomforest_regressor
from molehill.model import predict_randomforest_classifier, predict_randomforest_regressor
from molehill.model import _extract_attrs


@pytest.fixture
def categorical_cols():
    return ['cat1', 'cat2']


@pytest.fixture
def numerical_cols():
    return ['num1', 'num2', 'num3']


def test_extract_attrs(categorical_cols, numerical_cols):
    assert _extract_attrs(categorical_cols, numerical_cols) == '-attrs Q,Q,Q,C,C'
    assert _extract_attrs([], numerical_cols) == '-attrs Q,Q,Q'
    assert _extract_attrs(categorical_cols, []) == '-attrs C,C'


class TestSparseTrainModel:
    def test_train_randomforest_classifier(self):
        ret_sql = f"""\
-- molehill/{molehill.__version__}
with models as (
  select
    train_randomforest_classifier(
      features
      , target_val
      , '-trees 16'
    )
  from
    src_tbl
)
-- DIGDAG_INSERT_LINE
select
  model_id
  , model_weight
  , model
  , concat_ws(',', collect_set(concat(k1, ':', v1))) as var_importance
  , oob_errors
  , oob_tests
from
  models
lateral view explode(var_importance) t1 as k1, v1
group by 1, 2, 3, 5, 6
;
"""

        assert train_randomforest_classifier("src_tbl", "target_val", "-trees 16", sparse=True) == ret_sql

    def test_train_randomforest_regressor(self):
        ret_sql = f"""\
-- molehill/{molehill.__version__}
with models as (
  select
    train_randomforest_regressor(
      features
      , target_val
    )
  from
    src_tbl
)
-- DIGDAG_INSERT_LINE
select
  model_id
  , model_weight
  , model
  , concat_ws(',', collect_set(concat(k1, ':', v1))) as var_importance
  , oob_errors
  , oob_tests
from
  models
lateral view explode(var_importance) t1 as k1, v1
group by 1, 2, 3, 5, 6
;
"""

        assert train_randomforest_regressor("src_tbl", "target_val", sparse=True) == ret_sql


class TestDenseTrainModel:
    def test_train_randomforest_classifier(self, categorical_cols, numerical_cols):
        ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  train_randomforest_classifier(
    features
    , target_val
    , '-trees 16 -attrs Q,Q,Q,C,C'
  )
from
  src_tbl
;
"""

        assert train_randomforest_classifier(
            "src_tbl", "target_val", "-trees 16",
            categorical_columns=categorical_cols, numerical_columns=numerical_cols) == ret_sql

    def test_train_randomforest_regressor(self, categorical_cols, numerical_cols):
        ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  train_randomforest_regressor(
    features
    , target_val
    , '-attrs Q,Q,Q,C,C'
  )
from
  src_tbl
;
"""

        assert train_randomforest_regressor(
            "src_tbl", "target_val",
            categorical_columns=categorical_cols, numerical_columns=numerical_cols) == ret_sql


class TestPredictClassifier:
    def test_predict_randomforest_classifier(self):
        ret_sql = f"""\
-- molehill/{molehill.__version__}
with p as (
  select
    model_id
    , model_weight
    , model
  from
    model_tbl
  DISTRIBUTE BY rand(1)
),
t1 as (
  select
    t.id
    , p.model_weight
    , tree_predict(p.model_id, p.model, t.features, "-classification") as predicted
  from
    p
  left outer join target_tbl t
),
ensembled as (
  select
    id
    , rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  from
    t1
  group by
    id
)
-- DIGDAG_INSERT_LINE
select
  id
  , predicted.label
  , predicted.probabilities[1] as probability
from
  ensembled
;
"""

        pred_sql, pred_col = predict_randomforest_classifier("target_tbl", "id", "model_tbl")
        assert pred_sql == ret_sql
        assert pred_col == "probability"

    def test_predict_randomforest_classifier_hashing(self):
        ret_sql = f"""\
-- molehill/{molehill.__version__}
with p as (
  select
    model_id
    , model_weight
    , model
  from
    model_tbl
  DISTRIBUTE BY rand(1)
),
t1 as (
  select
    t.id
    , p.model_weight
    , tree_predict(p.model_id, p.model, feature_hashing(t.features), "-classification") as predicted
  from
    p
  left outer join target_tbl t
),
ensembled as (
  select
    id
    , rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  from
    t1
  group by
    id
)
-- DIGDAG_INSERT_LINE
select
  id
  , predicted.label
  , predicted.probabilities[1] as probability
from
  ensembled
;
"""
        pred_sql, pred_col = predict_randomforest_classifier("target_tbl", "id", "model_tbl", hashing=True)
        assert pred_sql == ret_sql
        assert pred_col == "probability"


class TestPredictRegressor:
    def test_predict_randomforest_regressor(self):
        ret_sql = f"""\
-- molehill/{molehill.__version__}
with p as (
  select
    model_id
    , model_weight
    , model
  from
    model_tbl
  DISTRIBUTE BY rand(1)
),
t1 as (
  select
    t.id
    , p.model_weight
    , tree_predict(p.model_id, p.model, t.features) as predicted
  from
    p
  left outer join target_tbl t
),
ensembled as (
  select
    id
    , rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  from
    t1
  group by
    id
)
-- DIGDAG_INSERT_LINE
select
  id
  , predicted.label
  , predicted.probabilities[1] as probability
from
  ensembled
;
"""
        pred_sql, pred_col = predict_randomforest_regressor("target_tbl", "id", "model_tbl")
        assert pred_sql == ret_sql
        assert pred_col == "target"

    def test_predict_regressor_hashing(self):
        ret_sql = f"""\
-- molehill/{molehill.__version__}
with p as (
  select
    model_id
    , model_weight
    , model
  from
    model_tbl
  DISTRIBUTE BY rand(1)
),
t1 as (
  select
    t.id
    , p.model_weight
    , tree_predict(p.model_id, p.model, feature_hashing(t.features)) as predicted
  from
    p
  left outer join target_tbl t
),
ensembled as (
  select
    id
    , rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  from
    t1
  group by
    id
)
-- DIGDAG_INSERT_LINE
select
  id
  , predicted.label
  , predicted.probabilities[1] as probability
from
  ensembled
;
"""
        pred_sql, pred_col = predict_randomforest_regressor("target_tbl", "id", "model_tbl", hashing=True)
        assert pred_sql == ret_sql
        assert pred_col == "target"

