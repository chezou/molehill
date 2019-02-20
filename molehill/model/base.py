import textwrap
from typing import Optional
from ..utils import build_query


def base_model(function: str,
               storage_format: Optional[str] = None,
               features: str = "features",
               target: str = "label",
               source_table: str = "training",
               option: Optional[str] = None,
               bias: Optional[bool] = None,
               hashing: Optional[bool] = None,
               with_clause: Optional[bool] = None) -> str:
    """Build model query

    Parameters
    -----------

    function : :obj:`str`
        A function name for algorithm.
    storage_format : :obj:`str`, optional
        Storage format. e.g. "feature, weight"
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

    _features = f"feature_hashing({features})" if hashing else features
    _features = f"add_bias({_features})" if bias else _features
    select_clause = textwrap.dedent("""\
    {function}(
      {features}
      , {target}
    """.format_map({"function": function, "features": _features, "target": target}))
    select_clause += f"  , '{option}'\n" if option else ""
    _as = f" as ({storage_format})" if storage_format else ''
    select_clause += f"){_as}"

    return build_query([select_clause], source_table, without_semicolon=with_clause)
