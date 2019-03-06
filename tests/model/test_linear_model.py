from molehill.model import train_classifier, train_regressor
from molehill.model import predict_classifier, predict_regressor


class TestTrainClassifier:
    def test_train_classifier(self):
        ret_sql = """\
select
  train_classifier(
    features
    , target_val
  ) as (feature, weight)
from
  src_tbl
;
"""

        assert train_classifier("src_tbl", "target_val") == ret_sql

    def test_train_classifier_pos_oversampling(self):
        ret_sql = """\
with train_oversampled as (
  select
    features
    , target_val
  from
    src_tbl
  where target_val = 0
  union all
  select
    features
    , target_val
  from
    (
      select
        amplify(${oversample_pos_n_times}, features, target_val) as (features, target_val)
      from
        src_tbl
      where target_val = 1
    ) t0
),
model_oversampled as (
  select
    train_classifier(
      features
      , target_val
    ) as (feature, weight)
  from
    train_oversampled
)
-- DIGDAG_INSERT_LINE
select
  feature
  , avg(weight) as weight
from
  model_oversampled
group by
  feature
;
"""

        assert train_classifier("src_tbl", "target_val", oversample_pos_n_times="${oversample_pos_n_times}") == ret_sql

    def test_train_classifier_oversampling(self):
        ret_sql = """\
with amplified as (
  select
    amplify(${oversample_n_times}, features, target_val) as (features, target_val)
  from
    src_tbl
),
train_oversampled as (
  select
    features
    , target_val
  from
    amplified
  CLUSTER BY rand(43)
),
model_oversampled as (
  select
    train_classifier(
      features
      , target_val
    ) as (feature, weight)
  from
    train_oversampled
)
-- DIGDAG_INSERT_LINE
select
  feature
  , avg(weight) as weight
from
  model_oversampled
group by
  feature
;
"""

        assert train_classifier("src_tbl", "target_val", oversample_n_times="${oversample_n_times}") == ret_sql

    def test_train_classifier_bias(self):
        ret_sql = """\
select
  train_classifier(
    add_bias(features)
    , target_val
  ) as (feature, weight)
from
  src_tbl
;
"""

        assert train_classifier("src_tbl", "target_val", bias=True) == ret_sql

    def test_train_classifier_hashing(self):
        ret_sql = """\
select
  train_classifier(
    feature_hashing(features)
    , target_val
  ) as (feature, weight)
from
  src_tbl
;
"""

        assert train_classifier("src_tbl", "target_val", hashing=True) == ret_sql

    def test_train_classifier_bias_hashing(self):
        ret_sql = """\
select
  train_classifier(
    add_bias(feature_hashing(features))
    , target_val
  ) as (feature, weight)
from
  src_tbl
;
"""

        assert train_classifier("src_tbl", "target_val", bias=True, hashing=True) == ret_sql


def test_train_regressor():
    ret_sql = """\
select
  train_regressor(
    features
    , target_val
  ) as (feature, weight)
from
  src_tbl
;
"""

    assert train_regressor("src_tbl", "target_val") == ret_sql


class TestPredictClassifier:
    def test_predict_classifier(self):
        ret_sql = """\
with features_exploded as (
  select
    id
    , extract_feature(fv) as feature
    , extract_weight(fv) as value
  from
    target_tbl t1
    LATERAL VIEW explode(features) t2 as fv
)
-- DIGDAG_INSERT_LINE
select
  t1.id
  , sigmoid(sum(m1.weight * t1.value)) as probability
from
  features_exploded t1
  left outer join model_tbl m1
    on (t1.feature = m1.feature)
group by
  t1.id
;
"""
        pred_sql, pred_col = predict_classifier("target_tbl", "id", "model_tbl")
        assert pred_sql == ret_sql
        assert pred_col == "probability"

    def test_predict_classifier_bias(self):
        ret_sql = """\
with features_exploded as (
  select
    id
    , extract_feature(fv) as feature
    , extract_weight(fv) as value
  from
    target_tbl t1
    LATERAL VIEW explode(add_bias(features)) t2 as fv
)
-- DIGDAG_INSERT_LINE
select
  t1.id
  , sigmoid(sum(m1.weight * t1.value)) as probability
from
  features_exploded t1
  left outer join model_tbl m1
    on (t1.feature = m1.feature)
group by
  t1.id
;
"""
        pred_sql, pred_col = predict_classifier("target_tbl", "id", "model_tbl", bias=True)
        assert pred_sql == ret_sql
        assert pred_col == "probability"

    def test_predict_classifier_hashing(self):
        ret_sql = """\
with features_exploded as (
  select
    id
    , extract_feature(fv) as feature
    , extract_weight(fv) as value
  from
    target_tbl t1
    LATERAL VIEW explode(feature_hashing(features)) t2 as fv
)
-- DIGDAG_INSERT_LINE
select
  t1.id
  , sigmoid(sum(m1.weight * t1.value)) as probability
from
  features_exploded t1
  left outer join model_tbl m1
    on (t1.feature = m1.feature)
group by
  t1.id
;
"""
        pred_sql, pred_col = predict_classifier("target_tbl", "id", "model_tbl", hashing=True)
        assert pred_sql == ret_sql
        assert pred_col == "probability"

    def test_predict_classifier_bias_hashing(self):
        ret_sql = """\
with features_exploded as (
  select
    id
    , extract_feature(fv) as feature
    , extract_weight(fv) as value
  from
    target_tbl t1
    LATERAL VIEW explode(add_bias(feature_hashing(features))) t2 as fv
)
-- DIGDAG_INSERT_LINE
select
  t1.id
  , sigmoid(sum(m1.weight * t1.value)) as probability
from
  features_exploded t1
  left outer join model_tbl m1
    on (t1.feature = m1.feature)
group by
  t1.id
;
"""
        pred_sql, pred_col = predict_classifier("target_tbl", "id", "model_tbl", bias=True, hashing=True)
        assert pred_sql == ret_sql
        assert pred_col == "probability"

    def test_predict_classifier_wo_sigmoid(self):
        ret_sql = """\
with features_exploded as (
  select
    id
    , extract_feature(fv) as feature
    , extract_weight(fv) as value
  from
    target_tbl t1
    LATERAL VIEW explode(features) t2 as fv
)
-- DIGDAG_INSERT_LINE
select
  t1.id
  , sum(m1.weight * t1.value) as total_weight
from
  features_exploded t1
  left outer join model_tbl m1
    on (t1.feature = m1.feature)
group by
  t1.id
;
"""
        pred_sql, pred_col = predict_classifier("target_tbl", "id", "model_tbl", sigmoid=False)
        assert pred_sql == ret_sql
        assert pred_col == "total_weight"


class TestPredictRegressor:
    def test_predict_regressor(self):
        ret_sql = """\
with features_exploded as (
  select
    id
    , extract_feature(fv) as feature
    , extract_weight(fv) as value
  from
    target_tbl t1
    LATERAL VIEW explode(features) t2 as fv
)
-- DIGDAG_INSERT_LINE
select
  t1.id
  , sum(m1.weight * t1.value) as target
from
  features_exploded t1
  left outer join model_tbl m1
    on (t1.feature = m1.feature)
group by
  t1.id
;
"""
        pred_sql, pred_col = predict_regressor("target_tbl", "id", "model_tbl", "target")
        assert pred_sql == ret_sql
        assert pred_col == "target"

    def test_predict_regressor_bias(self):
        ret_sql = """\
with features_exploded as (
  select
    id
    , extract_feature(fv) as feature
    , extract_weight(fv) as value
  from
    target_tbl t1
    LATERAL VIEW explode(add_bias(features)) t2 as fv
)
-- DIGDAG_INSERT_LINE
select
  t1.id
  , sum(m1.weight * t1.value) as target
from
  features_exploded t1
  left outer join model_tbl m1
    on (t1.feature = m1.feature)
group by
  t1.id
;
"""
        pred_sql, pred_col = predict_regressor("target_tbl", "id", "model_tbl", "target", bias=True)
        assert pred_sql == ret_sql
        assert pred_col == "target"

    def test_predict_regressor_hashing(self):
        ret_sql = """\
with features_exploded as (
  select
    id
    , extract_feature(fv) as feature
    , extract_weight(fv) as value
  from
    target_tbl t1
    LATERAL VIEW explode(feature_hashing(features)) t2 as fv
)
-- DIGDAG_INSERT_LINE
select
  t1.id
  , sum(m1.weight * t1.value) as target
from
  features_exploded t1
  left outer join model_tbl m1
    on (t1.feature = m1.feature)
group by
  t1.id
;
"""
        pred_sql, pred_col = predict_regressor("target_tbl", "id", "model_tbl", "target", hashing=True)
        assert pred_sql == ret_sql
        assert pred_col == "target"

    def test_predict_regressor_bias_hashing(self):
        ret_sql = """\
with features_exploded as (
  select
    id
    , extract_feature(fv) as feature
    , extract_weight(fv) as value
  from
    target_tbl t1
    LATERAL VIEW explode(add_bias(feature_hashing(features))) t2 as fv
)
-- DIGDAG_INSERT_LINE
select
  t1.id
  , sum(m1.weight * t1.value) as target
from
  features_exploded t1
  left outer join model_tbl m1
    on (t1.feature = m1.feature)
group by
  t1.id
;
"""
        pred_sql, pred_col = predict_regressor("target_tbl", "id", "model_tbl", "target", bias=True, hashing=True)
        assert pred_sql == ret_sql
        assert pred_col == "target"
