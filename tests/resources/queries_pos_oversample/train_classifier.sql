with train_oversampled as (
  select
    features
    , survived
  from
    ${source}
  where survived = 0
  union all
  select
    features
    , survived
  from
    (
      select
        amplify(${oversample_pos_n_times}, features, survived) as (features, survived)
      from
        ${source}
      where survived = 1
    ) t0
),
model_oversampled as (
  select
    train_classifier(
      features
      , survived
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
