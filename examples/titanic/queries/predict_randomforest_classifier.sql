with ensembled as (
  select
    rowid,
    rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
  from (
    select
      t.rowid,
      p.model_weight,
      tree_predict(p.model_id, p.model, t.features, "-classification") as predicted
    from (
      select
        model_id, model_weight, model
      from
        ${model_table}
      DISTRIBUTE BY rand(1)
    ) p
    left outer join ${target_table} t
  ) t1
  group by
    rowid
)
-- DIGDAG_INSERT_LINE
select
  rowid,
  predicted.label,
  predicted.probabilities[1] as probability
from
  ensembled
;
