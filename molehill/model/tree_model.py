from collections import OrderedDict
from typing import Optional, List, Tuple, Union
from .base import base_model
from ..utils import build_query


TREE_MODEL_TRAINERS = ['train_randomforest_classifier', 'train_randomforest_regressor']
TREE_MODEL_PREDICTORS = ['predict_randomforest_classifier', 'predict_randomforest_regressor']


def _extract_attrs(categorical_columns: List[str], numerical_columns: List[str]) -> str:
    attr_list = ['Q'] * len(numerical_columns) + ['C'] * len(categorical_columns)
    return f"-attrs {','.join(attr_list)}"


def _ensure_attrs(option: str, categorical_columns: List[str], numerical_columns: List[str]) -> str:
    has_attrs = any(_opt in option for _opt in ['-attrs', '-attribute_types'])

    if has_attrs:
        return option
    else:
        if len(option) > 0:
            option += ' '
        return f"{option}{_extract_attrs(categorical_columns, numerical_columns)}"


def _base_train_query(
        func_name: str,
        source_table: str,
        target: str,
        option: Optional[str],
        hashing: bool = False,
        oversample_pos_n_times: Optional[Union[int, str]] = None,
        sparse: bool = False,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None) -> str:

    if sparse:
        with_clause = base_model(
            func_name,
            None,
            target,
            source_table,
            option,
            hashing=hashing,
            with_clause=True,
            oversample_pos_n_times=oversample_pos_n_times)

        # Need to avoid Map format due to TD limitation.
        exploded_importance = "concat_ws(',', collect_set(concat(k1, ':', v1))) as var_importance"
        view_cond = "lateral view explode(var_importance) t1 as k1, v1\ngroup by 1, 2, 3, 5, 6"

        return build_query(["model_id", "model_weight", "model", exploded_importance, "oob_errors", "oob_tests"],
                           "models", view_cond, with_clauses=OrderedDict({"models": with_clause}))

    else:
        if categorical_columns is None and numerical_columns is None:
            raise ValueError("Either categorical_columns or numerical_columns should not be None")

        if categorical_columns is None:
            categorical_columns = []

        if numerical_columns is None:
            numerical_columns = []

        if option is None:
            option = ''

        option = _ensure_attrs(option, categorical_columns, numerical_columns)
        return base_model(
            func_name,
            None,
            target,
            source_table,
            option,
            hashing=hashing,
            oversample_pos_n_times=oversample_pos_n_times)


def train_randomforest_classifier(
        source_table: str = "${source}",
        target: str = "label",
        option: Optional[str] = None,
        hashing: bool = False,
        oversample_pos_n_times: Optional[Union[int, str]] = None,
        sparse: bool = False,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        **kwargs) -> str:
    """Build train_randomforest_classifier query

    Parameters
    -----------

    source_table : :obj:`str`
        Source table name. Default: "training"
    target : :obj:`str`
        Target column for prediction
    option : :obj:`str`, optional
        An option string for specific algorithm.
    hashing : bool
        Execute feature hashing. Default: False
    oversample_pos_n_times : int or :obj:`str`, optional
        Scale for oversampling positive class.
    sparse : bool
        Whether input vector is sparse or not. Default: False
    categorical_columns : :obj:`list` of :obj:`str`, optional
        List of categorical column names.
    numerical_columns : :obj:`list` of :obj:`str`, optional
        List of numerical column names.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    return _base_train_query(
        "train_randomforest_classifier",
        source_table,
        target,
        option=option,
        hashing=hashing,
        oversample_pos_n_times=oversample_pos_n_times,
        sparse=sparse,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns)


def train_randomforest_regressor(
        source_table: str = "${source}",
        target: str = "label",
        option: Optional[str] = None,
        hashing: bool = False,
        oversample_pos_n_times: Optional[Union[int, str]] = None,
        sparse: bool = False,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        **kwargs) -> str:
    """Build train_randomforest_classifier query

    Parameters
    -----------

    target : :obj:`str`
        Target column for prediction
    source_table : :obj:`str`
        Source table name. Default: "training"
    option : :obj:`str`, optional
        An option string for specific algorithm.
    hashing : bool
        Execute feature hashing. Default: False
    oversample_pos_n_times : int or :obj:`str`, optional
        Scale for oversampling positive class.
    sparse : bool
        Whether input vector is sparse or not. Default: False
    categorical_columns : :obj:`list` of :obj:`str`, optional
        List of categorical column names.
    numerical_columns : :obj:`list` of :obj:`str`, optional
        List of numerical column names.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    return _base_train_query(
        "train_randomforest_regressor",
        source_table,
        target,
        option=option,
        hashing=hashing,
        oversample_pos_n_times=oversample_pos_n_times,
        sparse=sparse,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns)


def _build_prediction_query(
        target_table: str,
        id_column: str,
        model_table: str,
        classification: bool = False,
        hashing: bool = False) -> str:

    _features = "t.features"
    _features = f"feature_hashing({_features})" if hashing else _features

    _with_clauses = OrderedDict()
    _with_clauses['p'] = build_query(
        ["model_id", "model_weight", "model"], model_table,
        condition="DISTRIBUTE BY rand(1)", without_semicolon=True)
    _classification = ', "-classification"' if classification else ''
    _with_clauses['t1'] = build_query(
        [f"t.{id_column}",
         "p.model_weight",
         f"tree_predict(p.model_id, p.model, {_features}{_classification}) as predicted"],
        "p",
        condition=f"left outer join {target_table} t", without_semicolon=True)
    _with_clauses['ensembled'] = build_query(
        [id_column, "rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted"],
        "t1",
        condition=f"group by\n  {id_column}", without_semicolon=True)
    query = build_query(
        [id_column, "predicted.label", "predicted.probabilities[1] as probability"],
        "ensembled", with_clauses=_with_clauses)
    return query


def predict_randomforest_classifier(
        target_table: str = "${target_table}",
        id_column: str = "rowid",
        model_table: str = "${model_table}",
        hashing: bool = False) -> Tuple[str, str]:
    """Build prediction query for randomforest classifier.

    Parameters
    ----------
    target_table : :obj:`str`, optional
        Target table name. Default: "${target_table}"
    id_column : :obj:`str`, optional
        Id column name. Default: "rowid"
    model_table : :obj:`str`
        Model table name.
    hashing : bool
        Execute feature hashing. Default: False

    Returns
    -------
    :obj:`str`
        Built query.
    :obj:`str`
        Predicted column name. For compatibility with predict_classifier.
    """

    return _build_prediction_query(target_table, id_column, model_table,
                                   classification=True, hashing=hashing), "probability"


def predict_randomforest_regressor(
        target_table: str = "${target_table}",
        id_column: str = "rowid",
        model_table: str = "${model_table}",
        hashing: bool = False) -> Tuple[str, str]:
    """Build prediction query for randomforest_regressor.

    Parameters
    ----------
    target_table : :obj:`str`, optional
        Target table name. Default: "${target_table}"
    id_column : :obj:`str`, optional
        Id column name. Default: "rowid"
    model_table : :obj:`str`
        Model table name.
    hashing : bool
        Execute feature hashing. Default: False

    Returns
    -------
    :obj:`str`
        Built query.
    :obj:`str`
        Predicted column name. For compatibility with predict_classifier.
    """

    return _build_prediction_query(target_table, id_column, model_table,
                                   hashing=hashing), "target"
