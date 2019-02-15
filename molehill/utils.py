import textwrap
from typing import List, Optional


def build_query(select_clauses: List[str],
                source: str,
                condition: Optional[str] = None,
                without_semicolon: Optional[bool] = None) -> str:
    """Build query from partial select clauses

    Parameters
    ----------
    select_clauses : :obj:`list` of :obj:`str`
        List of partial select clauses.
    source : :obj:`str`
        Source table name.
    condition : :obj:`str`, optional
        Condition like where clause.
    without_semicolon : bool, optional
        Whether put semicolon to end of the query or not.

    Returns
    -------
    :obj:`str`
        Complete query.

    Examples
    --------
    >>> from molehill.utils import build_query
    >>> select_clauses = ["col1 as a", "col2 as b"]
    >>> build_query(select_clauses, "sample_datasets")
    select
      col1 as a
      , col2 as b
    from
      sample_datasets
    ;
    """

    if not isinstance(select_clauses, list):
        raise ValueError("select_clauses must be list of str")

    query = "select\n"
    _query = ""
    _query += "\n, ".join(select_clauses)
    query += textwrap.indent(_query, "  ")

    query += textwrap.dedent("""
    from
      {source}""").format_map({"source": source})

    if condition:
        query += "\n" + condition

    if not without_semicolon:
        query += "\n;\n"

    return query
