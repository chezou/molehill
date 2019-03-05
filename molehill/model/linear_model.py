from typing import Optional, Tuple, Union
from .base import base_model
from ..utils import build_query


def train_classifier(
        source_table: str = "${source}",
        target: str = "target",
        option: Optional[str] = None,
        bias: bool = False,
        hashing: bool = False,
        scale_pos_weight: Optional[Union[int, str]] = None) -> str:
    """Build train_classifier query

    Parameters
    -----------

    source_table : :obj:`str`
        Source table name. Default: "training"
    target : :obj:`str`
        Target column for prediction. Default: "target"
    option : :obj:`str`
        An option string for specific algorithm.
    bias : bool
        Add bias or not. Default: False
    hashing : bool
        Execute feature hashing. Default: False
    scale_pos_weight : int or :obj:`str`, optional
        Scale for oversampling positive class.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    return base_model("train_classifier",
                      "feature, weight",
                      target,
                      source_table,
                      option,
                      bias=bias,
                      hashing=hashing,
                      scale_pos_weight=scale_pos_weight)


def train_regressor(
        source_table: str = "${source}",
        target: str = "target",
        option: Optional[str] = None,
        bias: bool = False,
        hashing: bool = False,
        scale_pos_weight: Optional[Union[int, str]] = None) -> str:
    """Build train_classifier query

    Parameters
    -----------

    target : :obj:`str`
        Target column for prediction. Default: "target"
    source_table : :obj:`str`
        Source table name. Default: "training"
    option : :obj:`str`, optional
        An option string for specific algorithm.
    bias : bool
        Add bias or not. Default: False
    hashing : bool
        Execute feature hashing. Default: False
    scale_pos_weight : int or :obj:`str`, optional
        Scale for oversampling positive class.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    return base_model("train_regressor",
                      "feature, weight",
                      target,
                      source_table,
                      option,
                      bias=bias,
                      hashing=hashing,
                      scale_pos_weight=scale_pos_weight)


def _build_prediction_query(
        predicted_column: str,
        target_table: str,
        id_column: str,
        model_table: str,
        bias: bool = False,
        hashing: bool = False,
        sigmoid: bool = False,
        oversampling: bool = False) -> str:

    _features = "features"
    _features = f"feature_hashing({_features})" if hashing else _features
    _features = f"add_bias({_features})" if bias else _features

    if sigmoid:
        _total_weight = f"sigmoid(sum(m1.weight * t1.value)) as {predicted_column}"
    else:
        _total_weight = f"sum(m1.weight * t1.value) as {predicted_column}"

    _with_clauses = {
        "features_exploded": build_query(
            [id_column, "extract_feature(fv) as feature", "extract_weight(fv) as value"],
            f"{target_table} t1\nLATERAL VIEW explode({_features}) t2 as fv",
            without_semicolon=True
        )
    }
    if oversampling:
        _with_clauses['score'] = build_query(
            [f"t1.{id_column}", _total_weight],
            f"features_exploded t1\nleft outer join {model_table} m1 on (t1.feature = m1.feature)",
            condition=f"group by \n  t1.{id_column}",
            without_semicolon=True)

        return build_query(
            [f"t.{id_column}",
             (f"t.{predicted_column} / (t.{predicted_column} + (1.0 - t.{predicted_column}) /"
              f" ${{td.last_results.downsampling_rate}}) as {predicted_column}")],
            "score t",
            with_clauses=_with_clauses)
    else:
        return build_query(
            [f"t1.{id_column}", _total_weight],
            f"features_exploded t1\nleft outer join {model_table} m1\n  on (t1.feature = m1.feature)",
            condition=f"group by\n  t1.{id_column}",
            with_clauses=_with_clauses)


def predict_classifier(
        target_table: str = "${target_table}",
        id_column: str = "rowid",
        model_table: str = "${model_table}",
        sigmoid: bool = True,
        bias: bool = False,
        hashing: bool = False,
        scale_pos_weight: Optional[Union[int, str]] = None) -> Tuple[str, str]:
    """Build a prediction query for train_classifier

    Parameters
    ----------
    target_table : :obj:`str`
        A table name for prediction.
    id_column : :obj:`str`
        ID column name.
    model_table : :obj:`str`
        A table name for trained model.
    sigmoid : bool
        Flag for using sigmoid or not. If you used logistic loss, sigmoid works fine.
        With this flag, the calculated column name will be probability, otherwise it'll be total_weight
        Default: True
    bias : bool
        Add bias or not. Default: False
    hashing : bool, optional
        Execute feature hashing. Default: False
    scale_pos_weight : int or :obj:`str`, optional
        Scale for oversampling positive class.

    Returns
    --------
    :obj:`str`
        Built query string.
    :obj:`str`
        Predicted column name. For compatibility with predict_classifier.
    """

    predicted_column = "probability" if sigmoid else "total_weight"

    return _build_prediction_query(
        predicted_column, target_table, id_column, model_table,
        bias=bias, hashing=hashing, sigmoid=sigmoid, oversampling=bool(scale_pos_weight)
    ), predicted_column


def predict_regressor(
        target_table: str = "${target_table}",
        id_column: str = "rowid",
        model_table: str = "${model_table}",
        predicted_column: str = "target",
        bias: bool = False,
        hashing: bool = False,
        scale_pos_weight: Optional[Union[int, str]] = None) -> Tuple[str, str]:
    """Build a prediction query for train_regressor

    Parameters
    ----------
    target_table : :obj:`str`
        A table name for prediction. Default: "${target_table}"
    id_column : :obj:`str`
        ID column name. Default: "rowid"
    model_table : :obj:`str`
        A table name for trained model. Default: "${model_table}"
    predicted_column : :obj:`str`
        A column name to store prediction results. Default: "target"
    bias : bool
        Add bias or not. Default: False
    hashing : bool
        Execute feature hashing. Default: False
    scale_pos_weight : int or :obj:`str`, optional
        Scale for oversampling positive class.

    Returns
    --------
    :obj:`str`
        Built query string.
    :obj:`str`
        Predicted column name. For compatibility with predict_classifier.
    """

    return _build_prediction_query(
        predicted_column, target_table, id_column, model_table,
        bias=bias, hashing=hashing, sigmoid=False, oversampling=bool(scale_pos_weight)
    ), predicted_column
