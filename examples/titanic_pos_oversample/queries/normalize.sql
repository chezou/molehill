-- client: molehill/0.0.1
select
  rowid
  , survived
  , rescale(
    age
    , ${td.last_results.age_min_train}
    , ${td.last_results.age_max_train}
  ) as age
  , rescale(
    fare
    , ${td.last_results.fare_min_train}
    , ${td.last_results.fare_max_train}
  ) as fare
  , embarked
  , sex
  , pclass
from
  ${source}
;
