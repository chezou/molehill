import molehill
from molehill.stats import compute_stats, combine_train_test_stats


def test_compute_stats():
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  avg(col1) as col1_mean
  , stddev_pop(col1) as col1_std
  , min(col1) as col1_min
  , approx_percentile(col1, 0.25) as col1_25
  , approx_percentile(col1, 0.5) as col1_median
  , approx_percentile(col1, 0.75) as col1_75
  , max(col1) as col1_max
  , avg(col2) as col2_mean
  , stddev_pop(col2) as col2_std
  , min(col2) as col2_min
  , approx_percentile(col2, 0.25) as col2_25
  , approx_percentile(col2, 0.5) as col2_median
  , approx_percentile(col2, 0.75) as col2_75
  , max(col2) as col2_max
from
  src_tbl
;
"""
    assert compute_stats('src_tbl', ['col1', 'col2']) == ret_sql


def test_combine_train_test_stats():
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  train.col1_mean as col1_mean_train
  , train.col1_std as col1_std_train
  , train.col1_min as col1_min_train
  , train.col1_25 as col1_25_train
  , train.col1_median as col1_median_train
  , train.col1_75 as col1_75_train
  , train.col1_max as col1_max_train
  , test.col1_mean as col1_mean_test
  , test.col1_std as col1_std_test
  , test.col1_min as col1_min_test
  , test.col1_25 as col1_25_test
  , test.col1_median as col1_median_test
  , test.col1_75 as col1_75_test
  , test.col1_max as col1_max_test
  , whole.col1_mean as col1_mean
  , whole.col1_std as col1_std
  , whole.col1_min as col1_min
  , whole.col1_25 as col1_25
  , whole.col1_median as col1_median
  , whole.col1_75 as col1_75
  , whole.col1_max as col1_max
from
  src_tbl_train_stats as train, src_tbl_test_stats as test, src_tbl_stats as whole
;
"""

    assert combine_train_test_stats('src_tbl', ['col1']) == ret_sql
