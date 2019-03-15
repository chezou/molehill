import textwrap
from typing import Union, List
from .utils import build_query


# metrics Hivemall supports
KNOWN_METRICS = {"logloss", "auc", "mse", "rmse", "mse", "mae", "r2", "fmeasure",
                 "average_precision", "hitrate", "ndcg", "precision_at", "recall_at"}
# metrics molehill extended
EXTENDED_METRICS = {"accuracy", "precision", "recall", "fmeasure_binary"}
PROBABILITY_REQUIRE_METRICS = {"logloss", "auc"}


def _build_evaluate_clause(
        metrics: List[str], scoring_template: str, inv_template: str, predicted_column: str, target_column: str):

    true_positive = f"sum(if({predicted_column} = {target_column} and {target_column} = 1, 1 , 0))"
    _results = []
    for _metric in metrics:
        if _metric == "fmeasure":
            _results.append(inv_template.format_map({
                "scoring": _metric, "predicted_column": predicted_column, "target_column": target_column, "option": ''
            }))
        elif _metric == "fmeasure_binary":
            _results.append(inv_template.format_map({
                "scoring": _metric, "predicted_column": predicted_column,
                "target_column": target_column, "option": ", '-average binary'"
            }))
        elif _metric == "auc":
            _results.append(scoring_template.format_map({
                "scoring": _metric, "predicted_column": "probability", "target_column": target_column
            }))
        elif _metric == "accuracy":
            _results.append(f"cast({true_positive} as double)/count(1) as accuracy")
        # precision and recall are only for binary class
        elif _metric == "precision":
            _results.append(f"cast({true_positive} as double)/sum(if({predicted_column} = 1, 1, 0)) as precision")
        elif _metric == "recall":
            _results.append(f"cast({true_positive} as double)/sum(if({target_column} = 1, 1, 0)) as precision")
        else:
            _results.append(scoring_template.format_map({
                "scoring": _metric, "predicted_column": predicted_column, "target_column": target_column
            }))

    return _results


def evaluate(
        metrics: Union[str, List[str]],
        target_column: str,
        predicted_column: str,
        target_table: str = "test",
        prediction_table: str = "prediction",
        id_column: str = "rowid") -> str:
    """Build evaluation query.

    Parameters
    ----------
    metrics : :obj:`str` or :obj:`list` of :obj:`str`
        Metrics for evaluation.
    prediction_table : :obj:`str`
        A table name for prediction results.
    predicted_column : :obj:`str`
        Predicted column name in a prediction table.
    target_table : :obj:`str`
        Test table name.
    target_column : :obj:`str`
        Target column name for actual value in a test table.
    id_column : :obj:`str`
        Id column name to join prediction and test table.

    Returns
    -------
    :obj:`str`
        Built query for evaluation.
    """

    _metrics = [metrics] if isinstance(metrics, str) else metrics
    _metrics = [metric.lower() for metric in metrics]

    if len(set(_metrics) - KNOWN_METRICS - EXTENDED_METRICS) > 0:
        unknown_metrics = [s for s in metrics if s not in KNOWN_METRICS]

        raise ValueError("Unknown metric: {}".format(", ".join(unknown_metrics)))

    has_auc = 'auc' in _metrics

    if has_auc:
        scoring_template = "{scoring}({predicted_column}, {target_column}) as {scoring}"
        inv_template = "{scoring}({target_column}, {predicted_column}{option}) as {scoring}"

        evaluations = _build_evaluate_clause(_metrics, scoring_template, inv_template, predicted_column, target_column)

        select_clause = f"p.{predicted_column}, t.{target_column}"

        cond = textwrap.dedent("""\
        join
          {test} t on (p.{id} = t.{id})
        order by
          probability desc""".format_map({"test": target_table, "id": id_column}))

        return build_query(
            evaluations,
            "(\n{}\n) t2".format(
                textwrap.indent(build_query(
                    [select_clause],
                    f"{prediction_table} p",
                    cond,
                    without_semicolon=True
                ), "  ")
            )
        )

    else:
        scoring_template = "{scoring}(p.{predicted_column}, t.{target_column}) as {scoring}"
        inv_template = "{scoring}(t.{target_column}, p.{predicted_column}) as {scoring}"

        # TODO: Handle option for scoring
        evaluations = _build_evaluate_clause(_metrics, scoring_template, inv_template, predicted_column, target_column)

        cond = textwrap.dedent(f"""\
        join
          {target_table} t on (p.{id_column} = t.{id_column})""")

        return build_query(evaluations, f"{prediction_table} p", cond)
