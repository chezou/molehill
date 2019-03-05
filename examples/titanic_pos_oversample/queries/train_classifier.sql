with train_oversampling as (
  select
    features
    , survived
  from
    ${source}
  where survived = 0
  union all
  select
    amplify(${scale_pos_weight}, features, survived) as (features, survived)
  from
    ${source}
  where survived = 1
)
-- DIGDAG_INSERT_LINE
select
  train_classifier(
    features
    , survived
  ) as (feature, weight)
from
  train_oversampling
;
