import textwrap
from typing import Optional
from .base import base_model


def train_classifier(
        features: str = "features",
        target: str = "target",
        source_table: str = "train",
        option: Optional[str] = None,
        bias: Optional[bool] = None) -> str:
    """Build train_classifier query

    Parameters
    -----------

    features :  :obj:`str`
        Feature column name. Default: "features"
    target : :obj:`str`
        Target column for prediction. Default: "target"
    source_table : :obj:`str`
        Source table name. Default: "training"
    option : :obj:`str`
        An option string for specific algorithm.
    bias : bool
        Add bias or not.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    return base_model("train_classifier",
                      "feature, weight",
                      features,
                      target,
                      source_table,
                      option,
                      bias)


def train_regressor(
        features: str = "features",
        source_table: str = "train",
        target: str = "target",
        option: Optional[str] = None,
        bias: Optional[bool] = None) -> str:
    """Build train_classifier query

    Parameters
    -----------

    features :  :obj:`str`
        Feature column name. Default: "features"
    target : :obj:`str`
        Target column for prediction. Default: "target"
    source_table : :obj:`str`
        Source table name. Default: "training"
    option : :obj:`str`
        An option string for specific algorithm.
    bias : bool
        Add bias or not.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    return base_model("train_regressor",
                      "feature, weight",
                      features,
                      target,
                      source_table,
                      option,
                      bias)


def _build_prediction_query(
        total_weight: str,
        target_table: str,
        id_column: str,
        features: str,
        model_table: str,
        model_feature: str) -> str:

    template = textwrap.dedent("""\
    with features_exploded as (
      select
        {id}
        , extract_feature(fv) as feature
        , extract_weight(fv) as value
      from
        {target_table} t1
        LATERAL VIEW explode({features}) t2 as fv
    )
    -- DIGDAG_INSERT_LINE
    select
      t1.{id}
      , {total_weight}
    from
      features_exploded t1
      left outer join {model_table} m1
        on (t1.feature = m1.{model_feature})
    group by
      t1.{id}
    ;
    """)

    return template.format_map({
        "id": id_column, "target_table": target_table, "features": features,
        "total_weight": total_weight, "model_table": model_table, "model_feature": model_feature
    })


def predict_classifier(
        target_table: str = "test",
        id_column: str = "rowid",
        features: str = "features",
        model_table: str = "model",
        model_weight: str = "weight",
        model_feature: str = "feature",
        sigmoid: Optional[bool] = True) -> str:
    """Build a prediction query for train_classifiere

    Parameters
    ----------
    target_table : :obj:`str`
        A table name for prediction.
    id_column : :obj:`str`
        ID column name.
    features : :obj:`str`
        Feature column name in the target table.
    model_table : :obj:`str`
        A table name for trained model.
    model_weight : :obj:`str`
        A column name for weight in a model table.
    model_feature : :obj:`str`
        A column name for feature in a model table.
    sigmoid : bool, optional
        Flag for using sigmoid or not. If you used logistic loss, sigmoid works fine.
        With this flag, the calculated column name will be probability, otherwise it'll be total_weight
        Default: True
    Returns
    --------
    :obj:`str`
    """

    if sigmoid:
        _total_weight = "sigmoid(sum(m1.{} * t1.value)) as probability".format(model_weight)
    else:
        _total_weight = "sum(m1.{} * t1.value) as total_weight".format(model_weight)

    return _build_prediction_query(
        _total_weight, target_table, id_column, features, model_table, model_feature)


def predict_regressor(
        target_table: str = "${source}_test",
        id_column: str = "rowid",
        features: str = "features",
        model_table: str = "model",
        model_weight: str = "weight",
        model_feature: str = "feature",
        predicted_column: str = "target") -> str:
    """Build a prediction query for train_regressor

    Parameters
    ----------
    target_table : :obj:`str`
        A table name for prediction.
    id_column : :obj:`str`
        ID column name.
    features : :obj:`str`
        Feature column name in the target table.
    model_table : :obj:`str`
        A table name for trained model.
    model_weight : :obj:`str`
        A column name for weight in a model table.
    model_feature : :obj:`str`
        A column name for feature in a model table.
    predicted_column : :obj:`str`
        A column name to store prediction results.
    Returns
    --------
    :obj:`str`
    """

    _total_weight = "sum(m1.{} * t1.value) as {}".format(model_weight, predicted_column)

    return _build_prediction_query(
        _total_weight, target_table, id_column, features, model_table, model_feature)
