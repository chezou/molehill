import pytest
from molehill.evaluation import evaluate


def test_evaluate_with_auc():
    metrics = ['auc', 'logloss']
    ret_sql = """\
select
  auc(predicted, target) as auc
  , logloss(predicted, target) as logloss
from
  (
    select
      p.predicted, t.target
    from
      pred p
    join
      actual t on (p.id = t.id)
    order by
      probability desc
  ) t2
;
"""
    assert evaluate(metrics, 'target', 'predicted', 'actual', 'pred', 'id') == ret_sql


def test_evaluate_without_auc():
    metrics = ['logloss']
    ret_sql = """\
select
  logloss(p.predicted, t.target) as logloss
from
  prediction p
join
  test t on (p.rowid = t.rowid)
;
"""
    assert evaluate(metrics, 'target', 'predicted') == ret_sql


def test_evaluate_fmeasure():
    metrics = ['fmeasure']
    ret_sql = """select
  fmeasure(t.target, p.predicted) as fmeasure
from
  prediction p
join
  test t on (p.rowid = t.rowid)
;
"""
    assert evaluate(metrics, 'target', 'predicted') == ret_sql


def test_evaluate_unknown_measure():
    metrics = ['unknown_metrics']
    with pytest.raises(ValueError):
        evaluate(metrics, 'target', 'predicted')
