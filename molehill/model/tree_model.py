import textwrap
from typing import Optional, List, Tuple
from .base import base_model
from ..utils import build_query


def extract_attrs(categorical_columns: List[str], numerical_columns: List[str]) -> str:
    attr_list = ['Q'] * len(numerical_columns) + ['C'] * len(categorical_columns)
    return f"-attrs {','.join(attr_list)}"


def _base_train_query(
        func_name: str,
        features: str,
        source_table: str,
        target: str,
        option: str,
        bias: bool,
        hashing: bool) -> str:

    with_clause = base_model(
        func_name,
        None,
        features,
        target,
        source_table,
        option,
        bias=bias,
        hashing=hashing,
        with_clause=True)

    # Need to avoid Map format due to TD limitation.
    exploded_importance = "concat_ws(',', collect_set(concat(k1, ':', v1))) as var_importance"
    view_cond = "lateral view explode(var_importance) t1 as k1, v1\ngroup by 1, 2, 3, 5, 6"

    return build_query(["model_id", "model_weight", "model", exploded_importance, "oob_errors", "oob_tests"],
                       "models", view_cond, with_clauses={"models": with_clause})


def train_randomforest_classifier(
        features: str = "features",
        source_table: str = "${source}",
        target: str = "label",
        option: Optional[str] = None,
        bias: Optional[bool] = None,
        hashing: Optional[bool] = None) -> str:
    """Build train_randomforest_classifier query

    Parameters
    -----------

    features :  :obj:`str`
        Feature column name. Default: "features"
    source_table : :obj:`str`
        Source table name. Default: "training"
    target : :obj:`str`
        Target column for prediction
    option : :obj:`str`
        An option string for specific algorithm.
    bias : bool
        Add bias or not.
    hashing : bool, optional
        Execute feature hashing.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    return _base_train_query(
        "train_randomforest_classifier",
        features,
        source_table,
        target,
        option,
        bias,
        hashing)


def train_randomforest_regressor(
        features: str = "features",
        source_table: str = "${source}",
        target: str = "label",
        option: Optional[str] = None,
        bias: Optional[bool] = None,
        hashing: Optional[bool] = None) -> str:
    """Build train_randomforest_classifier query

    Parameters
    -----------

    features :  :obj:`str`
        Feature column name. Default: "features"
    target : :obj:`str`
        Target column for prediction
    source_table : :obj:`str`
        Source table name. Default: "training"
    option : :obj:`str`
        An option string for specific algorithm.
    bias : bool
        Add bias or not.
    hashing : bool, optional
        Execute feature hashing.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    return _base_train_query(
        "train_randomforest_regressor",
        features,
        source_table,
        target,
        option,
        bias,
        hashing)


def _build_prediction_query(
        target_table: str,
        id_column: str,
        model_table: str,
        classification: Optional[bool],
        bias: Optional[bool] = None,
        hashing: Optional[bool] = None) -> str:

    _features = "t.features"
    _features = f"feature_hashing({_features})" if hashing else _features
    _features = f"add_bias({_features})" if bias else _features

    query = textwrap.dedent("""\
    with ensembled as (
      select
        {id},
        rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
      from (
        select
          t.{id}, 
          p.model_weight,
          tree_predict(p.model_id, p.model, {features}{classification}) as predicted
        from (
          select 
            model_id, model_weight, model
          from 
            {model_table} 
          DISTRIBUTE BY rand(1)
        ) p
        left outer join {target_table} t
      ) t1
      group by
        {id}
    )
    -- DIGDAG_INSERT_LINE
    select 
      {id},
      predicted.label
      predicted.probabilities[1] as probability
    from
      ensembled
    ;""").format_map({"id": id_column, "model_table": model_table, "target_table": target_table, "features": _features,
                      "classification": ' , "-classification"' if classification else ''})

    return query


def predict_randomforest_classifier(
        target_table: Optional[str] = "${target_table}",
        id_column: Optional[str] = "rowid",
        model_table: Optional[str] = "${model_table}",
        bias: Optional[bool] = None,
        hashing: Optional[bool] = None) -> Tuple[str, str]:
    """Build prediction query for randomforest classifier.

    Parameters
    ----------
    target_table : :obj:`str`, optional
        Target table name. Default: "${target_table}"
    id_column : :obj:`str`, optional
        Id column name. Default: "rowid"
    model_table : :obj:`str`
        Model table name.
    bias : bool
        Add bias or not.
    hashing : bool, optional
        Execute feature hashing.

    Returns
    -------
    :obj:`str`
        Built query.
    :obj:`str`
        Predicted column name. For compatibility with predict_classifier.
    """

    return _build_prediction_query(target_table, id_column, model_table,
                                   classification=True, bias=bias, hashing=hashing), "probability"


def predict_randomforest_regressor(
        target_table: str = "${target_table}",
        id_column: str = "rowid",
        model_table: str = "${model_table}",
        bias: Optional[bool] = None,
        hashing: Optional[bool] = None) -> Tuple[str, str]:
    """Build prediction query for randomforest regressor.

    Parameters
    ----------
    target_table : :obj:`str`, optional
        Target table name. Default: "${target_table}"
    id_column : :obj:`str`, optional
        Id column name. Default: "rowid"
    model_table : :obj:`str`
        Model table name.
    bias : bool
        Add bias or not.
    hashing : bool, optional
        Execute feature hashing.

    Returns
    -------
    :obj:`str`
        Built query.
    :obj:`str`
        Predicted column name. For compatibility with predict_classifier.
    """

    return _build_prediction_query(target_table, id_column, model_table,
                                   bias=bias, hashing=hashing), "target"
