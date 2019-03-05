import textwrap
from typing import List, Optional
from ..utils import build_query


FEATURE_FUNC_MAP = {"numerical": "quantitative", "categorical": "categorical"}


def vectorize(
        source: str,
        target_column: str,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        id_column: str = "rowid",
        features: str = "features",
        bias: bool = False,
        hashing: bool = False,
        emit_null: bool = False,
        force_value: bool = False) -> str:
    """Build vectorization query before training or prediction.

    Parameters
    ----------
    source : :obj:`str`
        Source table name.
    target_column : :obj:`str`
        Target column name for prediction.
    categorical_columns : :obj:`list` of :obj:`str`, optional
        A list of categorical column names.
    numerical_columns : :obj:`list` of :obj:`str`, optional
        A list of numerical column names.
    id_column : :obj:`str`
        Id column name. Default: "rowid"
    features : :obj:`str`
        Feature column name. Default: "features"
    bias : bool
        Add bias for feature. Default: False
    hashing : bool
        Execute feature hashing. Default: False
        If there is a large number of categorical features, hashing at vectorization phase would be better.
    emit_null : bool
        Ensure feature entity size equally with emitting Null or 0. Default: False
    force_value : bool
        Force to output value as 1 for categorical columns. Default: False

    Returns
    -------
    :obj:`str`
       Built query for vectorization.
    """

    if categorical_columns is None and numerical_columns is None:
        raise ValueError("Either one categorical or numerical column is required.")

    if not categorical_columns:
        categorical_columns = []

    if not numerical_columns:
        numerical_columns = []

    feature_query = _feature_column_query(
        categorical_columns, numerical_columns, emit_null=emit_null, force_value=force_value)

    feature_query = "feature_hashing(\n{}\n)".format(textwrap.indent(feature_query, "  ")) if hashing else feature_query
    feature_query = "add_bias(\n{}\n)".format(textwrap.indent(feature_query, "  ")) if bias else feature_query
    feature_query += f" as {features}"

    query = build_query(
        [id_column, feature_query, target_column],
        source
    )

    return query


def _build_feature_array(
        columns: List[str],
        ctype: str,
        emit_null: bool = False,
        force_value: bool = False) -> str:
    """ Build feature array for vectorization.

    Parameters
    -----------
    columns : :obj:`list` of :obj:`str`
        A list of column names
    ctype : :obj:`str`
        A type of column. "numerical" or "categorical" can be used.
    emit_null : bool
        Ensure output null or 0 if a categorical column is null or 0 value. Default: False
    force_value : bool
        Force to output value as 1 for categorical columns. Default: False

    Returns
    --------
    str
        Partial query for feature vectorization.
    """

    _query = ""
    _query += 'array("'
    _query += '", "'.join(columns)
    _query += '")\n, '

    _query += "\n, ".join(columns)

    _options = []
    if emit_null:
        _options.append('-emit_null')
    if force_value:
        _options.append('-force_value')

    _option = ""
    if len(_options) > 0:
        _option = ", '{}'\n".format(' '.join(_options))

    return "{func}_features(\n{query}\n{option})".format_map({
        "func": FEATURE_FUNC_MAP[ctype],
        "query": textwrap.indent(_query, "  "),
        "option": textwrap.indent(_option, "  ")
    })


def _feature_column_query(
        categorical_columns: List[str],
        numerical_columns: List[str],
        emit_null: bool = False,
        force_value: bool = False) -> str:
    """Build feature column query.

    Parameters
    ----------
    categorical_columns : :obj:`list` of :obj:`str`
        A list of categorical column names.
    numerical_columns : :obj:`list` of :obj:`str`
        A list of numerical column names.
    emit_null : bool
        Ensure output null or 0 if a categorical column is null or 0 value. Default: False
    force_value : bool
        Force to output value as 1 for categorical columns. Default: False

    Returns
    -------
    :obj:`str`
        Built feature column query

    """

    exists_numerical = len(numerical_columns) > 0
    exists_categorical = len(categorical_columns) > 0
    both_column_type = exists_numerical and exists_categorical

    _query = ""

    if both_column_type:
        _query = "array_concat(\n"

    if exists_numerical:
        feature_array = _build_feature_array(numerical_columns, "numerical", emit_null)
        _query += textwrap.indent(feature_array, "  ") if both_column_type else feature_array

    if both_column_type:
        _query += ",\n"

    if exists_categorical:
        feature_array = _build_feature_array(categorical_columns, "categorical", emit_null, force_value)
        _query += textwrap.indent(feature_array, "  ") if both_column_type else feature_array

    if both_column_type:
        _query += "\n)"

    return _query
