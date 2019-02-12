import textwrap
from typing import List, Optional
from ..utils import build_query


FEATURE_FUNC_MAP = {"numerical": "quantitative", "categorical": "categorical"}


def vectorize(
        source: str,
        target_column: str,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        id_column: Optional[str] = "rowid",
        features: Optional[str] = "features"):
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
    id_column : :obj:`str`, optional
        Id column name.
    features : :obj:`str`, optional
        Feature column name.
    Returns
    -------

    """

    if categorical_columns is None and numerical_columns is None:
        raise ValueError("Either one categorical or numerical column is required.")

    if not categorical_columns:
        categorical_columns = []

    if not numerical_columns:
        numerical_columns = []

    feature_query = _feature_column_query(categorical_columns, numerical_columns, features)
    query = build_query(
        [id_column, feature_query, target_column],
        source
    )

    return query


def _build_feature_array(columns: List[str], ctype: str) -> str:
    """ Build feature array for vectorization.

    Parameters
    -----------
    columns : :obj:`list` of :obj:`str`
        A list of column names
    ctype : :obj:`str`
        A type of column. "numerical" or "categorical" can be used.

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

    return "{}_features(\n{}\n)".format(FEATURE_FUNC_MAP[ctype], textwrap.indent(_query, "  "))


def _feature_column_query(
        categorical_columns: List[str],
        numerical_columns: List[str],
        features: str) -> str:
    """Build feature column query.

    Parameters
    ----------
    categorical_columns : :obj:`list` of :obj:`str`
        A list of categorical column names.
    numerical_columns : :obj:`list` of :obj:`str`
        A list of numerical column names.
    features : :obj:`str`
        Features column name.

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
        _query = "concat_array(\n"

    if exists_numerical:
        feature_array = _build_feature_array(numerical_columns, "numerical")
        _query += textwrap.indent(feature_array, "  ") if both_column_type else feature_array

    if both_column_type:
        _query += ",\n"

    if exists_categorical:
        feature_array = _build_feature_array(categorical_columns, "categorical")
        _query += textwrap.indent(feature_array, "  ") if both_column_type else feature_array

    if both_column_type:
        _query += "\n)"

    _query += " as {}".format(features)

    return _query
