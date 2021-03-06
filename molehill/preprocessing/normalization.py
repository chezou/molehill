import textwrap
from builtins import ValueError
from typing import List, Optional


class Normalizer:
    """Normalizing or Scaling numerical values.

    Examples
    --------
    >>> from molehill.utils import build_query
    >>> numeric_columns = ["age", "fare"]
    >>> numeric_normalizer = Normalizer("minmax", "train")
    >>>
    >>> # For transform
    >>> transform_clause = numeric_normalizer.transform(numeric_columns)
    >>> source = "titanic_train"
    >>> build_query([transform_clause], source)
    >>>
    >>> # For invert transform
    >>> inv_transform_clause = numeric_normalizer.invert_transform(numeric_columns)
    >>> build_query([inv_transform_clause], source)
    """

    def __init__(self, strategy: str = "log1p", phase: Optional[str] = "train") -> None:
        self.strategy = strategy
        self.phase = _phase = "_{}".format(phase) if phase else ""

    def _build_partial_query(self, template: str, _columns: List[str]) -> str:
        __query = "\n, ".join(
            template.format_map(
                {
                    "column": column,
                    "phase": self.phase,
                }
            )
            for column in _columns
        )
        return __query

    def transform(self, columns: List[str]) -> str:
        if self.strategy == "log1p":
            _template = "ln({column} + 1) as {column}"

        elif self.strategy == "minmax":
            _template = textwrap.dedent(
                """\
            rescale(
              {column}
              , ${{td.last_results.{column}_min{phase}}}
              , ${{td.last_results.{column}_max{phase}}}
            ) as {column}"""
            )

        elif self.strategy == "standardize":
            _template = textwrap.dedent(
                """\
            zscore(
              {column}
              , ${{td.last_results.{column}_mean{phase}}}
              , ${{td.last_results.{column}_std{phase}}}
            ) as {column}"""
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return self._build_partial_query(_template, columns)

    def invert_transform(self, columns: List[str]) -> str:
        _query = ""

        if self.strategy == "log1p":
            _template = "exp({column}) - 1 as {column}"

        elif self.strategy == "minmax":
            _template = textwrap.dedent(
                """\
            {column} * (${{td.last_results.{column}_max{phase}}} - ${{td.last_results.{column}_min{phase}}})
              + ${{td.last_results.{column}_min{phase}}} as {column}"""
            )

        elif self.strategy == "standardize":
            _template = textwrap.dedent(
                """\
            {column} * ${{td.last_results.{column}_std{phase}}} + ${{td.last_results.{column}_mean{phase}}} as {column}"""
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return self._build_partial_query(_template, columns)
