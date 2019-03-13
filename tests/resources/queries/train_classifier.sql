-- client: molehill/0.0.1
select
  train_classifier(
    features
    , survived
  ) as (feature, weight)
from
  ${source}
;
