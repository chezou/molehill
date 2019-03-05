from molehill.preprocessing import downsampling_rate


def test_downsampling_rate():
    ret_sql = """\
with label_count as (
  select
    label
    , cast(count(1) as double) as cnt
  from
    train
  group by
    label
),
aggregated as (
  select
    map_agg(label, cnt) as kv
  from
    label_count
)
-- DIGDAG_INSERT_LINE
select
  (kv[0] / (kv[0] + kv[1] * ${scale_pos_weight}.0)) / (kv[0] / (kv[0] + kv[1])) as downsampling_rate
from
  aggregated
;
"""

    assert downsampling_rate("train", "label") == ret_sql
