import textwrap


def build_query(select_clauses, source):
    """Build query from partial select clauses

    Parameters
    ----------
    select_clauses : list of str
        List of partial select clauses.
    source : str
        Source table name.

    Returns
    -------
    str
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

    if type(select_clauses) is not list:
        ValueError("select_clauses must be list of str")

    query = "select\n"
    _query = ""
    _query += "\n, ".join(select_clauses)
    query += textwrap.indent(_query, "  ")

    query += textwrap.dedent("""
    from
      {source}
    ;""").format_map({"source": source})

    return query
