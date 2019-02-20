from builtins import ValueError
from typing import List, Optional, Any


class Imputer:
    """Impute NULL value.

    Examples
    --------
    >>> from molehill.utils import build_query
    >>>
    >>> numeric_imputer = Imputer("median", "train")
    >>> categorical_imputer = Imputer("constant", "train", "missing")
    >>> numeric_columns = ["age", "fare"]
    >>> categorical_columns = ["embarked", "sex", "pclass"]
    >>> numeric_clause = numeric_imputer.transform(numeric_columns)
    >>> categorical_clause = categorical_imputer.transform(categorical_columns)
    >>> source = "titanic_train"
    >>> build_query([numeric_clause, categorical_clause], source)
    """

    def __init__(self,
                 strategy: str = "mean",
                 phase: Optional[str] = "train",
                 fill_value: Optional[Any] = None,
                 categorical: Optional[bool] = None) -> None:
        self.strategy = strategy
        self.phase = "_{}".format(phase) if phase else ""
        self.fill_value = "'{}'".format(fill_value) if type(fill_value) == str else fill_value
        self.categorical = categorical

    def _build_partial_query(self, template: str, statistics: str, _columns: List[str]) -> str:
        __query = "\n, ".join(
            template.format_map(
                {
                    "column": column,
                    "phase": self.phase,
                    "statistics": statistics,
                }
            )
            for column in _columns
        )
        return __query

    def transform(self, columns: List[str]) -> str:
        if self.strategy == "mean":
            statistics = "${{td.last_results.{column}_mean{phase}}}"

        elif self.strategy == "median":
            statistics = "${{td.last_results.{column}_median{phase}}}"

        elif self.strategy == "constant":
            if self.fill_value is None:
                raise ValueError("fill_value should not be None.")

            statistics = self.fill_value

        else:
            raise ValueError("strategy should be mean, median or constant")

        _column_source = "cast({column} as varchar)" if self.categorical else "{column}"
        _template = "coalesce({column}, {statistics}) as {column_dest}".format_map({
            "column": _column_source, "column_dest": "{column}", "statistics": statistics})

        return self._build_partial_query(_template, statistics, columns)
