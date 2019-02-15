import textwrap
from typing import Optional
from ..utils import build_query


def base_model(function: str,
               storage_format: str,
               features: str = "features",
               target: str = "label",
               source_table: str = "training",
               option: Optional[str] = None,
               bias: Optional[bool] = None
               ) -> str:
    """Build model query

    Parameters
    -----------

    function : :obj:`str`
        A function name for algorithm.
    storage_format : :obj:`str`
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

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    _features = "add_bias({})".format(features) if bias else features
    select_clause = textwrap.dedent("""\
    {function}(
      {features}
      , {target}
    """.format_map({"function": function, "features": _features, "target": target}))
    select_clause += "  , {}\n".format(option) if option else ""
    select_clause += ") as ({})".format(storage_format)

    return build_query([select_clause], source_table)
