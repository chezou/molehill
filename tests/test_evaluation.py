import pytest
import molehill
from molehill.evaluation import evaluate


def test_evaluate_with_auc():
    metrics = ['auc', 'logloss']
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  auc(probability, target) as auc
  , logloss(probability, target) as logloss
from
  (
    select
      p.probability, t.target
    from
      pred p
    join
      actual t on (p.id = t.id)
    order by
      probability desc
  ) t2
;
"""
    assert evaluate(metrics, 'target', 'probability', 'actual', 'pred', 'id') == ret_sql


def test_evaluate_with_logloss():
    metrics = ['logloss']
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  logloss(p.probability, t.target) as logloss
from
  prediction p
join
  test t on (p.rowid = t.rowid)
;
"""
    assert evaluate(metrics, 'target', 'predicted') == ret_sql


def test_evaluate_fmeasure():
    metrics = ['fmeasure']
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  fmeasure(t.target, p.predicted) as fmeasure
from
  prediction p
join
  test t on (p.rowid = t.rowid)
;
"""
    assert evaluate(metrics, 'target', 'predicted') == ret_sql


def test_evaluate_with_accuracy_precision_recall():
    metrics = ['accuracy', 'precision', 'recall']
    ret_sql = f"""\
-- molehill/{molehill.__version__}
select
  cast(sum(if(p.predicted = t.target and t.target = 1, 1, 0)) as double)/count(1) as accuracy
  , cast(sum(if(p.predicted = t.target and t.target = 1, 1, 0)) as double)/sum(if(p.predicted = 1, 1, 0)) as "precision"
  , cast(sum(if(p.predicted = t.target and t.target = 1, 1, 0)) as double)/sum(if(t.target = 1, 1, 0)) as recall
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
