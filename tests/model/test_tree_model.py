import pytest
from molehill.model import train_randomforest_classifier, train_randomforest_regressor
from molehill.model import predict_randomforest_classifier, predict_randomforest_regressor
from molehill.model import extract_attrs


@pytest.fixture
def categorical_cols():
    return ['cat1', 'cat2']


@pytest.fixture
def numerical_cols():
    return ['num1']


def test_extract_attrs():
    assert extract_attrs(['cat1', 'cat2'], ['num1', 'num2', 'num3']) == '-attrs Q,Q,Q,C,C'
    assert extract_attrs([], ['num1', 'num2', 'num3']) == '-attrs Q,Q,Q'
    assert extract_attrs(['cat1', 'cat2'], []) == '-attrs C,C'


class TestTrainClassifier:
    def test_train_randomforest_classifier_sparse(self):
        ret_sql = """\
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

    def test_train_randomforest_classifier_dense(self, categorical_cols, numerical_cols):
        ret_sql = """\
select
  train_randomforest_classifier(
    array(num1, cat1, cat2)
    , target_val
    , '-trees 16 -attrs Q,C,C'
  )
from
  src_tbl
;
"""

        assert train_randomforest_classifier(
            "src_tbl", "target_val", "-trees 16",
            categorical_columns=categorical_cols, numerical_columns=numerical_cols) == ret_sql

    def test_train_randomforest_classifier_dense_hash(self, categorical_cols, numerical_cols):
        ret_sql = """\
select
  train_randomforest_classifier(
    array(num1, mhash(cat1), mhash(cat2))
    , target_val
    , '-trees 16 -attrs Q,C,C'
  )
from
  src_tbl
;
"""

        assert train_randomforest_classifier(
            "src_tbl", "target_val", "-trees 16", hashing=True,
            categorical_columns=categorical_cols, numerical_columns=numerical_cols) == ret_sql

    def test_train_randomforest_classifier_null_columns(self, categorical_cols, numerical_cols):
        with pytest.raises(ValueError):
            train_randomforest_classifier(
                "src_tbl", "target_val", "-trees 16", hashing=True)

        ret_sql1 = """\
select
  train_randomforest_classifier(
    array(num1)
    , target_val
    , '-trees 16 -attrs Q'
  )
from
  src_tbl
;
"""

        assert train_randomforest_classifier(
            "src_tbl", "target_val", "-trees 16", hashing=True,
            numerical_columns=numerical_cols) == ret_sql1

        ret_sql2 = """\
select
  train_randomforest_classifier(
    array(mhash(cat1), mhash(cat2))
    , target_val
    , '-trees 16 -attrs C,C'
  )
from
  src_tbl
;
"""

        assert train_randomforest_classifier(
            "src_tbl", "target_val", "-trees 16", hashing=True,
            categorical_columns=categorical_cols) == ret_sql2


class TestTrainRegressor:
    def test_train_randomforest_regressor_sparse(self):
        ret_sql = """\
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

    def test_train_randomforest_regressor_dense(self, categorical_cols, numerical_cols):
        ret_sql = """\
select
  train_randomforest_regressor(
    array(num1, cat1, cat2)
    , target_val
    , '-trees 16 -attrs Q,C,C'
  )
from
  src_tbl
;
"""

        assert train_randomforest_regressor(
            "src_tbl", "target_val", "-trees 16",
            categorical_columns=categorical_cols, numerical_columns=numerical_cols) == ret_sql

    def test_train_randomforest_regressor_dense_hash(self, categorical_cols, numerical_cols):
        ret_sql = """\
select
  train_randomforest_regressor(
    array(num1, mhash(cat1), mhash(cat2))
    , target_val
    , '-trees 16 -attrs Q,C,C'
  )
from
  src_tbl
;
"""

        assert train_randomforest_regressor(
            "src_tbl", "target_val", "-trees 16", hashing=True,
            categorical_columns=categorical_cols, numerical_columns=numerical_cols) == ret_sql


class TestPredictClassifier:
    def test_predict_randomforest_classifier_sparse(self):
        ret_sql = """\
with ensembled as (
  select
    id,
    rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  from (
    select
      t.id,
      p.model_weight,
      tree_predict(p.model_id, p.model, t.features, "-classification") as predicted
    from (
      select
        model_id, model_weight, model
      from
        model_tbl
      DISTRIBUTE BY rand(1)
    ) p
    left outer join target_tbl t
  ) t1
  group by
    id
)
-- DIGDAG_INSERT_LINE
select
  id,
  predicted.label,
  predicted.probabilities[1] as probability
from
  ensembled
;
"""

        pred_sql, pred_col = predict_randomforest_classifier("target_tbl", "id", "model_tbl", sparse=True)
        assert pred_sql == ret_sql
        assert pred_col == "probability"

    def test_predict_randomforest_classifier_sparse_hashing(self):
        ret_sql = """\
with ensembled as (
  select
    id,
    rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  from (
    select
      t.id,
      p.model_weight,
      tree_predict(p.model_id, p.model, feature_hashing(t.features), "-classification") as predicted
    from (
      select
        model_id, model_weight, model
      from
        model_tbl
      DISTRIBUTE BY rand(1)
    ) p
    left outer join target_tbl t
  ) t1
  group by
    id
)
-- DIGDAG_INSERT_LINE
select
  id,
  predicted.label,
  predicted.probabilities[1] as probability
from
  ensembled
;
"""
        pred_sql, pred_col = predict_randomforest_classifier("target_tbl", "id", "model_tbl", hashing=True, sparse=True)
        assert pred_sql == ret_sql
        assert pred_col == "probability"

    def test_predict_randomforest_classifier_dense(self, numerical_cols, categorical_cols):
        ret_sql = """\
with ensembled as (
  select
    id,
    rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  from (
    select
      t.id,
      p.model_weight,
      tree_predict(p.model_id, p.model, array(t.num1, t.cat1, t.cat2), "-classification") as predicted
    from (
      select
        model_id, model_weight, model
      from
        model_tbl
      DISTRIBUTE BY rand(1)
    ) p
    left outer join target_tbl t
  ) t1
  group by
    id
)
-- DIGDAG_INSERT_LINE
select
  id,
  predicted.label,
  predicted.probabilities[1] as probability
from
  ensembled
;
"""

        pred_sql, pred_col = predict_randomforest_classifier(
            "target_tbl", "id", "model_tbl",
            categorical_columns=categorical_cols, numerical_columns=numerical_cols)
        assert pred_sql == ret_sql
        assert pred_col == "probability"

    def test_predict_randomforest_classifier_sparse_hashing(self):
        ret_sql = """\
with ensembled as (
  select
    id,
    rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  from (
    select
      t.id,
      p.model_weight,
      tree_predict(p.model_id, p.model, feature_hashing(t.features), "-classification") as predicted
    from (
      select
        model_id, model_weight, model
      from
        model_tbl
      DISTRIBUTE BY rand(1)
    ) p
    left outer join target_tbl t
  ) t1
  group by
    id
)
-- DIGDAG_INSERT_LINE
select
  id,
  predicted.label,
  predicted.probabilities[1] as probability
from
  ensembled
;
"""
        pred_sql, pred_col = predict_randomforest_classifier("target_tbl", "id", "model_tbl", hashing=True, sparse=True)
        assert pred_sql == ret_sql
        assert pred_col == "probability"

    def test_predict_randomforest_classifier_dense_hashing(self, categorical_cols, numerical_cols):
        ret_sql = """\
with ensembled as (
  select
    id,
    rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  from (
    select
      t.id,
      p.model_weight,
      tree_predict(p.model_id, p.model, array(t.num1, mhash(t.cat1), mhash(t.cat2)), "-classification") as predicted
    from (
      select
        model_id, model_weight, model
      from
        model_tbl
      DISTRIBUTE BY rand(1)
    ) p
    left outer join target_tbl t
  ) t1
  group by
    id
)
-- DIGDAG_INSERT_LINE
select
  id,
  predicted.label,
  predicted.probabilities[1] as probability
from
  ensembled
;
"""
        pred_sql, pred_col = predict_randomforest_classifier(
            "target_tbl", "id", "model_tbl", hashing=True,
            categorical_columns=categorical_cols, numerical_columns=numerical_cols)
        assert pred_sql == ret_sql
        assert pred_col == "probability"


class TestPredictRegressorSparse:
    def test_predict_randomforest_regressor(self):
        ret_sql = """\
with ensembled as (
  select
    id,
    rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  from (
    select
      t.id,
      p.model_weight,
      tree_predict(p.model_id, p.model, t.features) as predicted
    from (
      select
        model_id, model_weight, model
      from
        model_tbl
      DISTRIBUTE BY rand(1)
    ) p
    left outer join target_tbl t
  ) t1
  group by
    id
)
-- DIGDAG_INSERT_LINE
select
  id,
  predicted.label,
  predicted.probabilities[1] as probability
from
  ensembled
;
"""
        pred_sql, pred_col = predict_randomforest_regressor("target_tbl", "id", "model_tbl", sparse=True)
        assert pred_sql == ret_sql
        assert pred_col == "target"

    def test_predict_regressor_hashing(self):
        ret_sql = """\
with ensembled as (
  select
    id,
    rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  from (
    select
      t.id,
      p.model_weight,
      tree_predict(p.model_id, p.model, feature_hashing(t.features)) as predicted
    from (
      select
        model_id, model_weight, model
      from
        model_tbl
      DISTRIBUTE BY rand(1)
    ) p
    left outer join target_tbl t
  ) t1
  group by
    id
)
-- DIGDAG_INSERT_LINE
select
  id,
  predicted.label,
  predicted.probabilities[1] as probability
from
  ensembled
;
"""

        pred_sql, pred_col = predict_randomforest_regressor("target_tbl", "id", "model_tbl", hashing=True, sparse=True)
        assert pred_sql == ret_sql
        assert pred_col == "target"
