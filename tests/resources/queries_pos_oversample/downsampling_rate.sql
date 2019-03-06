with label_count as (
  select
    ${target_column}
    , cast(count(1) as double) as cnt
  from
    ${source}
  group by
    ${target_column}
),
aggregated as (
  select
    map_agg(${target_column}, cnt) as kv
  from
    label_count
)
-- DIGDAG_INSERT_LINE
select
  (kv[0] / (kv[0] + kv[1] * ${oversample_pos_n_times}.0)) / (kv[0] / (kv[0] + kv[1])) as downsampling_rate
from
  aggregated
;
