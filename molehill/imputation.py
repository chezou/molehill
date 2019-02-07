from builtins import ValueError


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

    def __init__(self, strategy="mean", phase="train", fill_value=None):
        self.strategy = strategy
        self.phase = "_{}".format(phase) if phase else ""
        self.fill_value = '"{}"'.format(fill_value) if type(fill_value) == str else fill_value

    def _build_partial_query(self, template, statistics, _columns):
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

    def transform(self, columns):
        if self.strategy == "mean":
            statistics = "td.last_results.{column}_mean{phase}"

        elif self.strategy == "median":
            statistics = "td.last_results.{column}_median{phase}"

        elif self.strategy == "constant":
            if self.fill_value is None:
                raise ValueError("fill_value should not be None.")

            statistics = self.fill_value

        else:
            raise ValueError("strategy should be mean, median or constant")

        _template = "coalesce({column}, {statistics}) as {column}"

        return self._build_partial_query(_template, statistics, columns)
