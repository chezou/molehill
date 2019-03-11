from typing import List
from ..utils import build_query


def cardinality(
        source: str,
        categorical_columns: List[str]) -> str:

    cols = [f"approx_distinct({column})" for column in categorical_columns]
    select_clause = f"array_max(array[{', '.join(cols)}]) as max_categorical_cardinality"

    return build_query([select_clause], source)
