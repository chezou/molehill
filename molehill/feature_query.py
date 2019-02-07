import textwrap
from collections import defaultdict
from .utils import build_query


def build_feature_array(columns, ctype="quantitative"):
    """ Build feature array for vectorization.

    Parameters
    -----------
    columns : list
        A list of column names, or with normalization and imputation strategy.
        list can contain string for column name (["col1", "col2"]) or dict
        ([{'normalize': 'minmax', 'impute': 'mean', 'column': 'pcol'}])
    ctype : string
        A type of column. "quantitative" or "categorical" can be used.

    Returns
    --------
    str
        Partial query for feature vectorization.
    """

    query = ""
    if columns[0].get('column'):
        t_cols = defaultdict(list)
        for d in columns:
            norm = d.get('normalize')
            imp = d.get('impute')
            fill_value = None
            if imp == "constant":
                fill_value = d.get('fill_value', 0.0)

            t_cols[(norm, imp, fill_value)].append(d['column'])

    else:
        _query = ""
        _query += 'array("'
        _query += '", "'.join(columns)
        _query += '"),\n'

        _query += ",\n".join(columns)

        query = "{}_features(\n{}\n)".format(ctype, textwrap.indent(_query, "  "))

    return query


def vectorize_table(source, feature_query, target_column):
    query = build_query(
        ["rowid", feature_query, "{} as target".format(target_column)],
        source
    )

    return query


def feature_column_query(categorical_columns, numerical_columns):
    exists_numerical = len(numerical_columns) > 0
    exists_categorical = len(categorical_columns) > 0
    both_column_type = exists_numerical and exists_categorical

    _query = ""

    if both_column_type:
        _query = "concat_array(\n"

    if exists_numerical:
        feature_array = build_feature_array(numerical_columns, ctype="quantitative")
        if both_column_type:
            _query += textwrap.indent(feature_array, "  ")
        else:
            _query += feature_array

    if both_column_type:
        _query += ",\n"

    if exists_categorical:
        feature_array = textwrap.indent(
            build_feature_array(categorical_columns, ctype="categorical"), "  "
        )
        if both_column_type:
            _query += textwrap.indent(feature_array, "  ")
        else:
            _query += feature_array

    if both_column_type:
        _query += "\n)"

    _query += " as features"

    return _query
