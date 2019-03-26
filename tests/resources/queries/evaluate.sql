-- molehill/0.0.1
select
  auc(probability, survived) as auc
  , logloss(probability, survived) as logloss
from
  (
    select
      p.${predicted_column}, t.survived
    from
      ${predicted_table} p
    join
      ${actual} t on (p.rowid = t.rowid)
    order by
      probability desc
  ) t2
;
