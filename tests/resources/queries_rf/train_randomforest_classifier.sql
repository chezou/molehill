with models as (
  select
    train_randomforest_classifier(
      feature_hashing(features)
      , survived
      , '-trees 15 -seed 31'
    )
  from
    ${source}
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
