import yaml
from collections import OrderedDict
from typing import Optional
from pathlib import Path
from .preprocessing import shuffle, train_test_split
from .preprocessing import Imputer, Normalizer
from .preprocessing import vectorize
from .model import train_classifier, predict_classifier
from .evaluation import evaluate
from .stats import compute_stats, combine_train_test_stats
from .utils import build_query


def _represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


yaml.add_representer(OrderedDict, _represent_odict)


class Pipeline(object):
    def __init__(self):
        self.comp_stats_task = None
        self.combine_stats_path = None
        self.vectorize_target_train = None
        self.vectorize_target_test = None

    @staticmethod
    def save_query(file_path: str, query: str) -> None:
        p = Path(file_path)
        #if p.exists():
        #    raise ValueError("File `{}` already exists.".format(file_path))

        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', encoding='utf-8') as f:
            f.write(query)

    def _build_task_with_stats(
            self, query_path, source: str, output_prefix: str, hive: Optional[bool] = None):

        od = OrderedDict

        source_train = self.vectorize_target_train
        source_test = self.vectorize_target_test
        self.vectorize_target_train = "{}_train".format(output_prefix)
        self.vectorize_target_test = "{}_test".format(output_prefix)

        _exec_train_task = od({
            "td>": str(query_path),
            "engine": "hive" if hive else "presto",
            "source": source_train,
            "create_table": self.vectorize_target_train
        })
        exec_tasks = od({
            "+compute_stats": od({
                "_parallel": True,
                # "+whole": od(self.comp_stats_task, **{"source": source}),
                "+train": od(self.comp_stats_task, **{"source": source_train}),
                "+test": od(self.comp_stats_task, **{"source": source_test})
            }),
            "+combine_train_test_stats": od({
                "td>": str(self.combine_stats_path),
                "engine": "presto",
                "store_last_results": True
            }),
            "+execute": od({
                "_parallel": True,
                "+train": _exec_train_task,
                "+test": od(_exec_train_task,
                            **{"source": source_test,
                               "create_table": self.vectorize_target_test})
            })
        })

        return exec_tasks

    def dump_pipeline(self, config_file: str) -> str:
        od = OrderedDict

        with open(config_file, "r") as f:
            config = yaml.load(f)

        source = config['source']
        dbname = config['dbname']

        id_col = config['id_column']
        target_col = config['target_column']
        train_sample_rate = config["train_sample_rate"]

        columns = []
        numerical_columns = []
        categorical_columns = []
        imputed_columns = []
        imputation_clauses = []
        normalized_columns = []
        normalization_clauses = []

        for column_type in ["numerical_columns", "categorical_columns"]:
            for cols in config.get(column_type, []):
                _columns = cols["columns"]
                columns.extend(_columns)
                if column_type == "numerical_columns":
                    numerical_columns.extend(cols["columns"])
                elif column_type == "categorical_columns":
                    categorical_columns.extend(cols["columns"])

                if cols.get("transformer"):
                    if cols['transformer'].get('imputer'):
                        imputed_columns.extend(_columns)
                        _opt = cols['transformer']['imputer']
                        imp = Imputer(_opt['strategy'],
                                      _opt.get('phase'),
                                      _opt.get('fill_value'),
                                      column_type == "categorical_columns")
                        imputation_clauses.extend([imp.transform(_columns)])

                    if cols['transformer'].get('normalizer'):
                        normalized_columns.extend(_columns)
                        _opt = cols['transformer']['normalizer']
                        norm = Normalizer(_opt['strategy'], _opt.get('phase'))
                        normalization_clauses.extend([norm.transform(_columns)])

        workflow = od()
        export = od()
        export["!include"] = "config/params.yml"
        export["source"] = source
        export["td"] = {"database": dbname, "engine": "hive", "train_sample_rate": train_sample_rate}
        workflow["_export"] = export

        query_dir = Path(config.get("query_dir", "queries"))

        preparation = od()

        stratify = config.get("stratify")
        shuffle_query = shuffle(columns, target_column=target_col, id_column=id_col, stratify=stratify, class_label=target_col)
        shuffle_path = query_dir / "shuffle.sql"
        self.save_query(shuffle_path, shuffle_query)

        preparation["+shuffle"] = od({
            "td>": str(shuffle_path),
            "create_table": "${source}_shuffled"
        })

        train_query, test_query = train_test_split(stratify=stratify)
        split_train_path = query_dir / "split_train.sql"
        split_test_path = query_dir / "split_test.sql"
        self.save_query(split_train_path, train_query)
        self.save_query(split_test_path, test_query)

        preparation["+split"] = od({
            "_parallel": True,
            "+train": od({
                "td>": str(split_train_path),
                "engine": "presto",
                "create_table": "${source}_train"
            }),
            "+test": od({
                "td>": str(split_test_path),
                "engine": "presto",
                "create_table": "${source}_test"
            })
        })

        do_imputation = len(imputation_clauses) > 0
        do_normalization = len(normalization_clauses) > 0

        self.vectorize_target_train = "{}_train".format(source)
        self.vectorize_target_test = "{}_test".format(source)

        if do_imputation or do_normalization:
            stats_query = compute_stats("${source}", numerical_columns)
            combined_stats = combine_train_test_stats("${source}", numerical_columns)
            stats_path = query_dir / "stats.sql"
            self.combine_stats_path = query_dir / "combine_stats.sql"
            self.save_query(stats_path, stats_query)
            self.save_query(self.combine_stats_path, combined_stats)

            self.comp_stats_task = od({
                "td>": str(stats_path),
                "engine": "presto",
                "source": source,
                "create_table": "${source}_stats"})

        if do_imputation:
            non_imputed_columns = list(set(columns) - set(imputed_columns))
            imputation_clauses.extend(non_imputed_columns)
            imputation_query = build_query([id_col, target_col] + imputation_clauses, "${source}")
            impute_path = query_dir / "impute.sql"
            self.save_query(impute_path, imputation_query)

            preparation["+imputation"] = self._build_task_with_stats(
                impute_path, source, "{}_imputed".format(source))

        if do_normalization:
            non_normalized_columns = list(set(columns) - set(normalized_columns))
            normalization_clauses.extend(non_normalized_columns)
            normalization_query = build_query([id_col, target_col] + normalization_clauses, "${source}")
            normalize_path = query_dir / "normalize.sql"
            self.save_query(normalize_path, normalization_query)

            preparation["+normalization"] = self._build_task_with_stats(
                normalize_path, "{}_imputed".format(source), "{}_norm".format(source), hive=True)

        workflow["+preparation"] = preparation

        vectorize_query = vectorize("${source}", target_col, categorical_columns, numerical_columns, id_col)
        vectorize_path = query_dir / "vectorize.sql"
        self.save_query(vectorize_path, vectorize_query)

        train_table = "train"
        test_table = "test"
        vectorize_task = od({
            "_parallel": True,
            "+train": od({
                "td>": str(vectorize_path),
                "source": self.vectorize_target_train,
                "create_table": train_table
            }),
            "+test": od({
                "td>": str(vectorize_path),
                "source": self.vectorize_target_test,
                "create_table": test_table
            })
        })
        workflow["+vectorization"] = vectorize_task

        main = od()
        # TODO: Use config settings
        train_query = train_classifier(target=target_col, source_table=train_table)
        self.save_query(query_dir / "train.sql", train_query)
        model_table = "model"
        main["+train"] = od({
                "td>": str(query_dir / "train.sql"),
                "create_table": model_table
            })

        # TODO: Use config settings
        predict_query = predict_classifier("${source}", id_col, model_table=model_table)
        self.save_query(query_dir / "predict.sql", predict_query)
        predict_table = "prediction"
        main["+predict"] = od({
                "td>": str(query_dir / "predict.sql"),
                "source": test_table,
                "create_table": predict_table
            })

        metrics = config['evaluator']['metrics']
        _eval_target = config['evaluator'].get('target_table')

        evaluate_target = _eval_target if _eval_target else predict_table
        evaluate_query = evaluate(metrics,
                                  target_column="survived",
                                  target_table="${actual}",
                                  prediction_table="${predicted}",
                                  predicted_column="probability")
        self.save_query(query_dir / "evaluate.sql", evaluate_query)
        main["+evaluate"] = od({
                "td>": str(query_dir / "evaluate.sql"),
                "actual": test_table,
                "predicted": evaluate_target,
                "store_last_results": True
            })

        acc_template = "{metric}: ${{td.last_results.{metric}}}"
        acc_str = "\t".join(acc_template.format_map({"metric": metric}) for metric in metrics)
        main["+show_accuracy"] = {"echo>": acc_str}

        workflow["+main"] = main

        self.save_query("{}.dig".format(source), yaml.dump(workflow, default_flow_style=False))
