import molehill
from molehill.utils import build_query


def test_build_query():
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  col1
  , col2
from
  sample_datasets
;
"""
    assert build_query(['col1', 'col2'], 'sample_datasets') == ret_sql


def test_build_query_without_semicolon():
    ret_sql = f"""\
select
  col1
  , col2
from
  sample_datasets"""
    assert build_query(['col1', 'col2'], 'sample_datasets', without_semicolon=True) == ret_sql


def test_build_query_with_condition():
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  col1
  , col2
from
  sample_datasets
where
  sample = train
;
"""
    cond = "where\n  sample = train"
    assert build_query(['col1', 'col2'], 'sample_datasets', condition=cond) == ret_sql


def test_build_query_with_clause():
    ret_sql = f"""\
-- molehill/{molehill.__version__}
with test as (
  select
    col3
  from
    other
)
-- DIGDAG_INSERT_LINE
select
  col1
  , col2
from
  sample_datasets
;
"""
    with_clause = """\
select
  col3
from
  other"""
    assert build_query(['col1', 'col2'], 'sample_datasets', with_clauses={'test': with_clause}) == ret_sql
