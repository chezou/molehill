from ..utils import build_query


def downsampling_rate(source: str, target_column: str):
    _with_clauses = dict()
    _with_clauses['label_count'] = build_query(
        [target_column, "cast(count(1) as double) as cnt"], source,
        condition=f"group by\n  {target_column}", without_semicolon=True)
    _with_clauses['aggregated'] = build_query(
        [f"map_agg({target_column}, cnt) as kv"], "label_count", without_semicolon=True
    )

    return build_query(
        ["(kv[0] / (kv[0] + kv[1] * ${scale_pos_weight}.0)) / (kv[0] / (kv[0] + kv[1])) as downsampling_rate"],
        "aggregated", with_clauses=_with_clauses)
