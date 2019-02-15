import textwrap
import itertools
from typing import List
from .utils import build_query


def compute_stats(source: str, numerical_columns: List[str]) -> str:
    numerical_template = textwrap.dedent(
        """\
    avg({column}) as {column}_mean
    , stddev_pop({column}) as {column}_std
    , min({column}) as {column}_min
    , approx_percentile({column}, 0.25) as {column}_25
    , approx_percentile({column}, 0.5) as {column}_median
    , approx_percentile({column}, 0.75) as {column}_75
    , max({column}) as {column}_max"""
    )

    _query = ""
    _query += "\n, ".join(
        numerical_template.format_map({"column": column})
        for column in numerical_columns
    )

    return build_query([_query], source)


def combine_train_test_stats(source: str, numerical_columns: List[str]) -> str:
    numerical_template = textwrap.dedent(
        """\
    {phase}.{column}_mean as {column}_mean_{phase}
    , {phase}.{column}_std as {column}_std_{phase}
    , {phase}.{column}_min as {column}_min_{phase}
    , {phase}.{column}_25 as {column}_25_{phase}
    , {phase}.{column}_median as {column}_median_{phase}
    , {phase}.{column}_75 as {column}_75_{phase}
    , {phase}.{column}_max as {column}_max_{phase}"""
    )

    _query = ""
    _query += "\n, ".join(
        numerical_template.format_map({"column": column, "phase": phase})
        for column, phase in itertools.product(numerical_columns, ["train", "test"])
    )

    _source = "{source}_train_stats as train, {source}_test_stats as test".format_map(
        {"source": source})
    return build_query([_query], _source)
