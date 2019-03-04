import textwrap
from typing import Optional, Union
from ..utils import build_query


def base_model(function: str,
               storage_format: Optional[str] = None,
               target: str = "label",
               source_table: str = "training",
               option: Optional[str] = None,
               bias: Optional[bool] = None,
               hashing: Optional[bool] = None,
               with_clause: Optional[bool] = None,
               scale_pos_weight: Optional[Union[int, str]] = None) -> str:
    """Build model query

    Parameters
    -----------

    function : :obj:`str`
        A function name for algorithm.
    storage_format : :obj:`str`, optional
        Storage format. e.g. "feature, weight"
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
    with_clause : bool, optional
        Existence of with clause.
    scale_pos_weight : int, optional
        Scale for oversampling positive class.

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    _source_table = source_table
    _with_clauses = {}
    if scale_pos_weight:
        _with_clause = build_query(
            ["features", target],
            source_table,
            condition=f"where {target} = 0",
            without_semicolon=True)
        _with_clause += "\nunion all\n"
        _with_clause += build_query(
            [f"amplify(${{scale_pos_weight}}, features, {target}) as (features, {target})"],
            source_table,
            condition=f"where {target} = 1",
            without_semicolon=True)
        _source_table = "train_oversampling"
        _with_clauses = {_source_table: _with_clause}

    _features = "features"
    _features = f"feature_hashing({_features})" if hashing else _features
    _features = f"add_bias({_features})" if bias else _features

    select_clause = textwrap.dedent("""\
    {function}(
      {features}
      , {target}
    """.format_map({"function": function, "features": _features, "target": target}))
    select_clause += f"  , '{option}'\n" if option else ""
    _as = f" as ({storage_format})" if storage_format else ""
    select_clause += f"){_as}"

    return build_query([select_clause], _source_table, without_semicolon=with_clause, with_clauses=_with_clauses)
