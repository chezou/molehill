from molehill.preprocessing import shuffle, train_test_split


def test_shuffle():
    ret_sql = """\
select
  rowid() as id
  , target
  , col1
  , col2
  , rand(32) as rnd
from
  src_tbl
cluster by rand(43)
;
"""

    assert shuffle(['col1', 'col2'], 'target', 'src_tbl', 'id') == ret_sql


def test_stratified_shuffle():
    ret_sql = """\
select
  rowid() as id
  , target
  , col1
  , col2
  , count(1) over (partition by target) as per_label_count
  , rank() over (partition by target order by rand(32)) as rank_in_label
from
  src_tbl
;
"""

    assert shuffle(['col1', 'col2'], 'target', 'src_tbl', 'id', stratify=True) == ret_sql


def test_train_test_split():
    train_sql = """\
select
  *
from
  src_tbl
where
  rnd <= 0.8
;
"""
    test_sql = """\
select
  *
from
  src_tbl
where
  rnd > 0.8
;
"""
    gen_train, gen_test = train_test_split('src_tbl', 0.8)
    assert gen_train == train_sql
    assert gen_test == test_sql


def test_train_test_split_stratify():
    train_sql = """\
select
  *
from
  src_tbl
where
  rank_in_label <= (per_label_count * 0.8)
;
"""
    test_sql = """\
select
  *
from
  src_tbl
where
  rank_in_label > (per_label_count * 0.8)
;
"""
    gen_train, gen_test = train_test_split('src_tbl', 0.8, stratify=True)
    assert gen_train == train_sql
    assert gen_test == test_sql
