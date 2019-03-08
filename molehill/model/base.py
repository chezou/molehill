import textwrap
from collections import OrderedDict
from typing import Optional, Union
from ..utils import build_query


def base_model(function: str,
               storage_format: Optional[str] = None,
               target: str = "label",
               source_table: str = "training",
               option: Optional[str] = None,
               bias: bool = False,
               hashing: bool = False,
               with_clause: bool = False,
               oversample_pos_n_times: Optional[Union[int, str]] = None,
               oversample_n_times: Optional[Union[int, str]] = None,
               features: str = "features") -> str:
    """Build model query

    Parameters
    -----------

    function : :obj:`str`
        A function name for algorithm.
    storage_format : :obj:`str`, optional
        Storage format. e.g. "feature, weight"
    target : :obj:`str`
        Target column for prediction. Default: "label"
    source_table : :obj:`str`
        Source table name. Default: "training"
    option : :obj:`str`, optional
        An option string for specific algorithm.
    bias : bool
        Add bias or not. Default: False
    hashing : bool
        Execute feature hashing. Default: False
    with_clause : bool
        Existence of with clause. Default: False
    oversample_pos_n_times : int or :obj:`str`, optional
        Scale for oversampling positive class. This option and oversample_n_times are exclusive.
    oversample_n_times : int or :obj:`str`, optional
        Scale for oversampling train data. This option and oversample_pos_n_times are exclusive.
    features : :obj:`str`
        A string represents features. Default: "features"

    Returns
    --------
    :obj:`str`
        Built query for training.
    """

    _source_table = source_table
    _with_clauses = OrderedDict()  # type: OrderedDict[str, str]
    _without_semicolon = with_clause

    if oversample_pos_n_times and oversample_n_times:
        raise ValueError("scale_pos_weigh and oversample_n_times are exclusive.")

    if oversample_pos_n_times or oversample_n_times:
        _source_table = "train_oversampled"
        _without_semicolon = True

    _features = features
    _features = f"feature_hashing({_features})" if hashing else _features
    _features = f"add_bias({_features})" if bias else _features

    select_clause = textwrap.dedent(f"""\
    {function}(
      {_features}
      , {target}
    """)
    select_clause += f"  , '{option}'\n" if option else ''
    _as = f" as ({storage_format})" if storage_format else ''
    select_clause += f"){_as}"

    _query = build_query([select_clause], _source_table, without_semicolon=_without_semicolon)  # type: str

    if not oversample_pos_n_times and not oversample_n_times:
        return _query

    if oversample_pos_n_times:
        _with_clause = build_query(
            [features, target],
            source_table,
            condition=f"where {target} = 0",
            without_semicolon=True)
        _with_clause += "\nunion all\n"

        _oversample_query = build_query(
            [f"amplify({oversample_pos_n_times}, {features}, {target}) as (features, {target})"],
            source_table,
            condition=f"where {target} = 1",
            without_semicolon=True)
        _with_clause += build_query(
            [features, target],
            f"(\n{textwrap.indent(_oversample_query, '  ')}\n) t0",
            without_semicolon=True)
        _with_clauses["train_oversampled"] = _with_clause
        _with_clauses["model_oversampled"] = _query

    elif oversample_n_times:
        _with_clauses["amplified"] = build_query(
            [f"amplify({oversample_n_times}, {features}, {target}) as (features, {target})"],
            source_table,
            without_semicolon=True)

        _with_clauses["train_oversampled"] = build_query(
            [features, target],
            "amplified",
            condition="CLUSTER BY rand(43)",
            without_semicolon=True)

        _with_clauses["model_oversampled"] = _query

    return build_query(
        ["feature", "avg(weight) as weight"],
        "model_oversampled",
        condition="group by\n  feature",
        without_semicolon=with_clause, with_clauses=_with_clauses)
