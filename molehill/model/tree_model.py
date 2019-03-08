import textwrap
from collections import OrderedDict
from typing import Optional, List, Tuple, Union
from .base import base_model
from ..utils import build_query


TREE_TRAIN_MODELS = ['train_randomforest_classifier', 'train_randomforest_regressor']
TREE_PREDICT_MODELS = ['predict_randomforest_classifier', 'predict_randomforest_regressor']


def extract_attrs(categorical_columns: List[str], numerical_columns: List[str]) -> str:
    attr_list = ['Q'] * len(numerical_columns) + ['C'] * len(categorical_columns)
    return f"-attrs {','.join(attr_list)}"


def _build_features(
        numerical_columns: List[str], categorical_columns: List[str],
        hashing: bool = False, table: Optional[str] = None):

    if hashing:
        if table:
            _target_columns = [f"{table}.{e}" for e in numerical_columns] + \
                              [f"mhash({table}.{e})" for e in categorical_columns]

        else:
            _target_columns = numerical_columns + [f"mhash({e})" for e in categorical_columns]

    else:
        _target_columns = numerical_columns + categorical_columns

        if table:
            _target_columns = [f"{table}.{e}" for e in _target_columns]

    return f"array({', '.join(_target_columns)})"


def _base_train_query(
        func_name: str,
        source_table: str,
        target: str,
        option: Optional[str],
        bias: bool = False,
        hashing: bool = False,
        oversample_pos_n_times: Optional[Union[int, str]] = None) -> str:

    with_clause = base_model(
        func_name,
        None,
        target,
        source_table,
        option,
        bias=bias,
        hashing=hashing,
        with_clause=True,
        oversample_pos_n_times=oversample_pos_n_times)

    # Need to avoid Map format due to TD limitation.
    exploded_importance = "concat_ws(',', collect_set(concat(k1, ':', v1))) as var_importance"
    view_cond = "lateral view explode(var_importance) t1 as k1, v1\ngroup by 1, 2, 3, 5, 6"

    return build_query(["model_id", "model_weight", "model", exploded_importance, "oob_errors", "oob_tests"],
                       "models", view_cond, with_clauses=OrderedDict({"models": with_clause}))


def _base_train_query_with_dense(
        func_name: str,
        source_table: str,
        target: str,
        categorical_columns: List[str],
        numerical_columns: List[str],
        option: Optional[str],
        hashing: bool = False,
        oversample_pos_n_times: Optional[Union[int, str]] = None,
        **kwargs) -> str:

    _features = _build_features(numerical_columns, categorical_columns, hashing)

    if option is None or not any(_opt in option for _opt in ["-attrs", "-attribute_types"]):
        _options = [option] if option else []
        _options.append(extract_attrs(categorical_columns, numerical_columns))
        option = ' '.join(_options)

    return base_model(
        func_name,
        None,
        target,
        source_table,
        option,
        oversample_pos_n_times=oversample_pos_n_times,
        features=_features)


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
        Whether input is sparse vector or not.
        The algorithm for sparse vector is different from dense one so that it should be applied only when
        feature column size is not fixed.
    categorical_columns : :obj:`list` of :obj:`str`, optional
        Categorical column names to be converted dense vector.
    numerical_columns : :obj:`list` of :obj:`str`, optional
        Numerical column names to be converted dense vector.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    if sparse:
        return _base_train_query(
            "train_randomforest_classifier",
            source_table,
            target,
            option=option,
            hashing=hashing,
            oversample_pos_n_times=oversample_pos_n_times)

    else:
        if categorical_columns is None and numerical_columns is None:
            raise ValueError("Either categorical_columns or numerical_columns should be a list.")

        categorical_columns = [] if categorical_columns is None else categorical_columns
        numerical_columns = [] if numerical_columns is None else numerical_columns

        return _base_train_query_with_dense(
            "train_randomforest_classifier",
            source_table,
            target,
            categorical_columns,
            numerical_columns,
            option=option,
            hashing=hashing,
            oversample_pos_n_times=oversample_pos_n_times
        )


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
        Whether input is sparse vector or not. The algorithm for sparse vector is defferent from dense one.
    categorical_columns : :obj:`list` of :obj:`str`, optional
        Categorical column names to be converted dense vector.
    numerical_columns : :obj:`list` of :obj:`str`, optional
        Numerical column names to be converted dense vector.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    if sparse:
        return _base_train_query(
            "train_randomforest_regressor",
            source_table,
            target,
            option=option,
            hashing=hashing,
            oversample_pos_n_times=oversample_pos_n_times)

    else:
        if categorical_columns is None and numerical_columns is None:
            raise ValueError("Either categorical_columns or numerical_columns should be a list.")

        categorical_columns = [] if categorical_columns is None else categorical_columns
        numerical_columns = [] if numerical_columns is None else numerical_columns

        return _base_train_query_with_dense(
            "train_randomforest_regressor",
            source_table,
            target,
            categorical_columns,
            numerical_columns,
            option=option,
            hashing=hashing,
            oversample_pos_n_times=oversample_pos_n_times
        )


def _build_prediction_query(
        target_table: str,
        id_column: str,
        model_table: str,
        classification: bool = False,
        hashing: bool = False,
        features: str = "t.features") -> str:

    _features = features
    _features = f"feature_hashing({_features})" if hashing else _features

    query = textwrap.dedent(f"""\
    with ensembled as (
      select
        {id_column},
        rf_ensemble(predicted.value, predicted.posteriori, model_weight) as predicted
      from (
        select
          t.{id_column},
          p.model_weight,
          tree_predict(p.model_id, p.model, {_features}{', "-classification"' if classification else ''}) as predicted
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
        {id_column}
    )
    -- DIGDAG_INSERT_LINE
    select
      {id_column},
      predicted.label,
      predicted.probabilities[1] as probability
    from
      ensembled
    ;
    """)

    return query


def predict_randomforest_classifier(
        target_table: str = "${target_table}",
        id_column: str = "rowid",
        model_table: str = "${model_table}",
        hashing: bool = False,
        sparse: bool = False,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        **kwargs) -> Tuple[str, str]:
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
    sparse : bool
        Whether input is sparse vector or not. The algorithm for sparse vector is defferent from dense one.
    categorical_columns : :obj:`list` of :obj:`str`, optional
        Categorical column names to be converted dense vector.
    numerical_columns : :obj:`list` of :obj:`str`, optional
        Numerical column names to be converted dense vector.

    Returns
    -------
    :obj:`str`
        Built query.
    :obj:`str`
        Predicted column name. For compatibility with predict_classifier.
    """

    if sparse:
        return _build_prediction_query(
            target_table, id_column, model_table, classification=True, hashing=hashing), "probability"

    else:
        if categorical_columns is None and numerical_columns is None:
            raise ValueError("Either categorical_columns or numerical_columns should be a list.")

        categorical_columns = [] if categorical_columns is None else categorical_columns
        numerical_columns = [] if numerical_columns is None else numerical_columns

        features = _build_features(numerical_columns, categorical_columns, hashing, table="t")

        return _build_prediction_query(
            target_table, id_column, model_table, classification=True, features=features), "probability"


def predict_randomforest_regressor(
        target_table: str = "${target_table}",
        id_column: str = "rowid",
        model_table: str = "${model_table}",
        hashing: bool = False,
        sparse: bool = False,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        **kwargs) -> Tuple[str, str]:
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
    sparse : bool
        Whether input is sparse vector or not. The algorithm for sparse vector is defferent from dense one.
    categorical_columns : :obj:`list` of :obj:`str`, optional
        Categorical column names to be converted dense vector.
    numerical_columns : :obj:`list` of :obj:`str`, optional
        Numerical column names to be converted dense vector.

    Returns
    -------
    :obj:`str`
        Built query.
    :obj:`str`
        Predicted column name. For compatibility with predict_classifier.
    """

    if sparse:
        return _build_prediction_query(
            target_table, id_column, model_table, hashing=hashing), "target"

    else:
        if categorical_columns is None and numerical_columns is None:
            raise ValueError("Either categorical_columns or numerical_columns should be a list.")

        categorical_columns = [] if categorical_columns is None else categorical_columns
        numerical_columns = [] if numerical_columns is None else numerical_columns

        features = _build_features(numerical_columns, categorical_columns, hashing, table="t")

        return _build_prediction_query(
            target_table, id_column, model_table, features=features), "target"

