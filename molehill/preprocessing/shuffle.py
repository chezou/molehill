from typing import List, Union, Tuple, Optional
from ..utils import build_query


def shuffle(columns: List[str],
            target_column: str,
            source: str = "${source}",
            id_column: str = "rowid",
            stratify: Optional[bool] = None,
            rnd_seed: Optional[int] = 32,
            cluster_seed: Optional[int] = 43) -> str:
    """Build shuffle query for random sampling. Should be executed by Hive

    Parameters
    -----------
    columns : :obj:`list` of :obj:`str`
        List of column names.
    target_column : :obj:`str`
        Target label name for classification. It should be set with stratify option.
    source : :obj:`str`, optional
        Source table name. default "${source}".
    id_column: :obj:`str`, optional
        Id column name. default "rowid"
    stratify : bool, optional
        Flag for using stratified sampling.
        This flag requires class_label option and should ally with train_test_split function's option.
    rnd_seed : int, optional
        Random seed for random number for sampling.
    cluster_seed : int, optional
        Random seed for cluster by. Required for ordinal random shuffling.

    Returns
    --------
    :obj:`str`
        Shuffle query
    """

    _columns = [f"rowid() as {id_column}", target_column] + columns
    cond = ""

    if stratify:
        _columns.extend([
            f"count(1) over (partition by {target_column}) as per_label_count",
            f"rank() over (partition by {target_column} order by rand({rnd_seed})) as rank_in_label"
        ])
    else:
        _columns.extend([f"rand({rnd_seed}) as rnd"])
        cond = f"cluster by rand({cluster_seed})"

    return build_query(_columns, source, cond)


def train_test_split(source: str = "${source}_shuffled",
                     train_sample_rate: Union[int, str] = "${train_sample_rate}",
                     stratify: Optional[bool] = None) -> Tuple[str, str]:
    """Build train test split query. Should be executed by Hive

    Parameters
    -----------
    source : :obj:`str`
        Original source table name. default "${source}_shuffled"
    train_sample_rate : int or :obj:`str`
        Split ratio for train and test split. The value should be ratio of train examples.
        default "${train_sample_rate}"
    stratify : bool, optional
        If not None, data is split in a stratified fashion.

    Returns
    --------
    :obj:`tuple` of :obj:`str`
        Shuffle query for training and test data.
    """

    if stratify:
        condition_template = "where\n  rank_in_label {op} (per_label_count * {train_sample_rate})"
    else:
        condition_template = "where\n  rnd {op} {train_sample_rate}"

    train_query = build_query(
        ['*'],
        source,
        condition_template.format_map({
            "op": "<=", "train_sample_rate": train_sample_rate})
    )

    test_query = build_query(
        ['*'],
        source,
        condition_template.format_map({
            "op": ">", "train_sample_rate": train_sample_rate})
    )

    return train_query, test_query
