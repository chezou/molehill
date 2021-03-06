import textwrap
import molehill
from collections import OrderedDict
from typing import List, Optional


def build_query(select_clauses: List[str],
                source: str,
                condition: Optional[str] = None,
                without_semicolon: bool = False,
                with_clauses: Optional[OrderedDict] = None) -> str:
    """Build query from partial select clauses

    Parameters
    ----------
    select_clauses : :obj:`list` of :obj:`str`
        List of partial select clauses.
    source : :obj:`str`
        Source table name.
    condition : :obj:`str`, optional
        Condition like where clause.
    without_semicolon : bool
        Whether put semicolon to end of the query or not. Default: False
        It also suppresses header.
    with_clauses : :obj:`dict`, optional
        Key is a temporary table name and value is a with clause.

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

    query = ""

    # Add header
    if not without_semicolon:
        query += f"-- client: molehill/{molehill.__version__}\n"

    if not with_clauses:
        with_clauses = OrderedDict()

    _with_clauses = []

    for k, v in with_clauses.items():
        tmp = f"""\
{k} as (
{textwrap.indent(v, '  ')}
)"""
        _with_clauses.append(tmp)

    if len(with_clauses) > 0:
        query += "with {_with}\n-- DIGDAG_INSERT_LINE\n".format(_with=',\n'.join(_with_clauses))

    query += "select\n"
    _query = ""
    _query += "\n, ".join(select_clauses)
    query += textwrap.indent(_query, "  ")

    query += f"""
from
{textwrap.indent(source, '  ')}"""

    if condition:
        query += f"\n{condition}"

    if not without_semicolon or not len(with_clauses) == 0:
        query += "\n;\n"

    return query
