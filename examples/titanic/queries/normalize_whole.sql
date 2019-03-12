-- molehill/0.0.1
select
  rowid
  , survived
  , rescale(
    age
    , ${td.last_results.age_min}
    , ${td.last_results.age_max}
  ) as age
  , rescale(
    fare
    , ${td.last_results.fare_min}
    , ${td.last_results.fare_max}
  ) as fare
  , embarked
  , sex
  , pclass
from
  ${source}
;
