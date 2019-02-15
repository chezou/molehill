import textwrap
from typing import Union, List
from .utils import build_query


KNOWN_METRICS = {"logloss", "auc", "mse", "rmse", "mse", "mae", "r2",
                 "average_precision", "hitrate", "ndcg", "precision_at", "recall_at"}


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

    if len(set(_metrics) - KNOWN_METRICS) > 0:
        unknown_metrics = [s for s in metrics if s not in KNOWN_METRICS]

        raise ValueError("Unknown metric: {}".format(", ".join(unknown_metrics)))

    has_auc = 'auc' in _metrics

    if has_auc:
        scoring_template = "{scoring}({predicted_column}, {target_column}) as {scoring}"
        select_template = "p.{predicted_column}, t.{target_column}"
        inv_template = "{scoring}({target_column}, {predicted_column}) as {scoring}"

        evaluations = [
            inv_template.format_map({
                "scoring": _metric, "predicted_column": predicted_column, "target_column": target_column
            }) if _metric == "fmeasure"
            else scoring_template.format_map({
                "scoring": _metric, "predicted_column": predicted_column, "target_column": target_column
            })
            for _metric in _metrics
        ]

        select_clause = select_template.format_map({
                        "predicted_column": predicted_column,
                        "target_column": target_column})

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
                    "{} p".format(prediction_table),
                    cond,
                    without_semicolon=True
                ), "  ")
            )
        )

    else:
        scoring_template = "{scoring}(p.{predicted_column}, t.{target_column}) as {scoring}"
        inv_template = "{scoring}(t.{target_column}, p.{predicted_column}) as {scoring}"

        # TODO: Handle option for scoring
        evaluations = [
            inv_template.format_map({
                "scoring": _metric, "predicted_column": predicted_column, "target_column": target_column
            }) if _metric == "fmeasure"
            else scoring_template.format_map({
                "scoring": _metric, "predicted_column": predicted_column, "target_column": target_column
            })
            for _metric in _metrics
        ]

        cond = """\
        join
          {test} t on (p.{id} = t.{id})""".format_map({"test": target_table, "id": id_column})

        return build_query(evaluations, "{} p".format(prediction_table), textwrap.dedent(cond))
