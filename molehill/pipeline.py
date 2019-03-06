import sys
import shutil
import yaml
from collections import OrderedDict
from typing import Optional, List, Tuple, Any, Dict, Union
from pathlib import Path
from .preprocessing import shuffle, train_test_split
from .preprocessing import Imputer, Normalizer
from .preprocessing import vectorize
from .preprocessing import downsampling_rate
from .evaluation import evaluate
from .stats import compute_stats, combine_train_test_stats
from .utils import build_query


def _represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


def construct_odict(loader, node):
    return OrderedDict(loader.construct_pairs(node))


yaml.add_representer(OrderedDict, _represent_odict)
yaml.add_constructor('tag:yaml.org,2002:map', construct_odict)

od = OrderedDict


class Pipeline:
    def __init__(self):
        self.comp_stats_task = None
        self.combine_stats_path = None
        self.vectorize_target_train = None
        self.vectorize_target_test = None
        self.workflow_path = None
        self.query_dir = None
        self.columns = []
        self.numerical_columns = []
        self.categorical_columns = []
        self.imputed_columns = []
        self.imputation_clauses = []
        self.imputation_clauses_whole = []
        self.normalized_columns = []
        self.normalization_clauses = []
        self.normalization_clauses_whole = []
        self.id_column = None
        self.target_column = None

    @staticmethod
    def save_query(file_path: Union[str, Path], query: str) -> None:
        p = Path(file_path)

        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', encoding='utf-8') as f:
            f.write(query)

    def _set_columns(self, config: Dict[str, Any]) -> None:
        """Set columns, numerical_columns, categorical_columns from config. It also set transformation clauses

        Parameters
        ----------
        config : :obj:`Dict`
            Configuration dictionary including numerical_columns and categorical_columns key.
        Returns
        -------
        None
        """
        for column_type in ["numerical_columns", "categorical_columns"]:
            for cols in config.get(column_type, []):
                _columns = cols["columns"]
                self.columns.extend(_columns)
                if column_type == "numerical_columns":
                    self.numerical_columns.extend(cols["columns"])
                elif column_type == "categorical_columns":
                    self.categorical_columns.extend(cols["columns"])

                if cols.get("transformer"):
                    if cols['transformer'].get('imputer'):
                        self.imputed_columns.extend(_columns)
                        _opt = cols['transformer']['imputer']
                        imp = Imputer(_opt['strategy'],
                                      _opt.get('phase'),
                                      _opt.get('fill_value'),
                                      column_type == "categorical_columns")
                        self.imputation_clauses.extend([imp.transform(_columns)])
                        imp_whole = Imputer(_opt['strategy'],
                                            None,
                                            _opt.get('fill_value'),
                                            column_type == "categorical_columns")
                        self.imputation_clauses_whole.extend([imp_whole.transform(_columns)])

                    if cols['transformer'].get('normalizer'):
                        self.normalized_columns.extend(_columns)
                        _opt = cols['transformer']['normalizer']
                        norm = Normalizer(_opt['strategy'], _opt.get('phase'))
                        self.normalization_clauses.extend([norm.transform(_columns)])
                        norm_whole = Normalizer(_opt['strategy'], None)
                        self.normalization_clauses_whole.extend([norm_whole.transform(_columns)])

    def _build_shuffle_and_split_task(
            self,
            stratify: Optional[bool] = None) -> OrderedDict:
        preparation = od()  # type: OrderedDict[str, Any]

        shuffle_query = shuffle(
            self.columns, target_column=self.target_column, id_column=self.id_column, stratify=stratify)
        shuffle_path = self.query_dir / "shuffle.sql"
        self.save_query(shuffle_path, shuffle_query)

        preparation["+shuffle"] = od({
            "td>": str(shuffle_path),
            "create_table": "${source}_shuffled"
        })

        train_query, test_query = train_test_split(stratify=stratify)
        split_train_path = self.query_dir / "split_train.sql"
        split_test_path = self.query_dir / "split_test.sql"
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

        return preparation

    def _build_task_with_stats(
            self,
            query_basename: str,
            source: str,
            source_whole: Optional[str],
            output_prefix: str,
            target_columns: List[str],
            target_clauses: List[str],
            target_clauses_whole: List[str],
            hive: Optional[bool] = None) -> Tuple[OrderedDict, str, str]:

        query_path = str(self.query_dir / f"{query_basename}.sql")
        query_path_whole = str(self.query_dir / f"{query_basename}_whole.sql")
        complement_columns = []
        for _col in self.columns:
            if _col not in target_columns:
                complement_columns.append(_col)
        _target_clauses = target_clauses.copy()
        _target_clauses.extend(complement_columns)
        _target_clauses_whole = target_clauses_whole.copy()
        _target_clauses_whole.extend(complement_columns)

        transform_query = build_query(
            [self.id_column, self.target_column] + _target_clauses, "${source}")
        transform_query_whole = build_query(
            [self.id_column, self.target_column] + _target_clauses_whole, "${source}")

        self.save_query(query_path, transform_query)
        self.save_query(query_path_whole, transform_query_whole)

        source_train = f"{source}_train"
        source_test = f"{source}_test"
        vectorize_target_train = f"{output_prefix}_train"
        vectorize_target_test = f"{output_prefix}_test"

        _exec_train_task = od({
            "td>": query_path,
            "engine": "hive" if hive else "presto",
            "source": source_train,
            "create_table": vectorize_target_train
        })
        exec_tasks = od({
            "+compute_stats": od({
                "_parallel": True,
                "+whole": od(self.comp_stats_task, **{"source": source_whole}),
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
                "+whole": od(_exec_train_task,
                             **{"td>": query_path_whole,
                                "source": source_whole,
                                "create_table": output_prefix}),
                "+train": _exec_train_task,
                "+test": od(_exec_train_task,
                            **{"source": source_test,
                               "create_table": vectorize_target_test})
            })
        })

        return exec_tasks, vectorize_target_train, vectorize_target_test

    def _build_vectorize_task(
            self,
            conf: Dict[str, Any],
            source: str,
            source_train: str,
            source_test: str) -> Tuple[OrderedDict, str, str]:

        train_table = conf.pop("train_table", "train")
        test_table = conf.pop("test_table", "test")
        whole_table = conf.pop("whole_table", "whole")

        vect_default_opt = {"categorical_columns": self.categorical_columns,
                            "numerical_columns": self.numerical_columns,
                            "id_column": self.id_column}
        vectorize_query = vectorize("${source}", self.target_column, **dict(vect_default_opt, **conf))
        vectorize_path = self.query_dir / "vectorize.sql"
        self.save_query(vectorize_path, vectorize_query)

        vectorize_task = od({
            "_parallel": True,
            "+whole": od({
                "td>": str(vectorize_path),
                "source": source,
                "create_table": whole_table
            }),
            "+train": od({
                "td>": str(vectorize_path),
                "source": source_train,
                "create_table": train_table
            }),
            "+test": od({
                "td>": str(vectorize_path),
                "source": source_test,
                "create_table": test_table
            })
        })

        return vectorize_task, train_table, test_table

    def _build_train_task(
            self,
            config: Dict[str, Any],
            mod: object,
            train_table: str) -> OrderedDict:

        func_name = config.pop('name')
        train_func = getattr(mod, func_name)

        if func_name == 'train_randomforest_classifier' and config.pop('guess_attrs', None):
            from .model import extract_attrs
            _opt = config.get('option', '')
            _opt += ' ' + extract_attrs(self.categorical_columns, self.numerical_columns)
            config['option'] = _opt

        model_table = config.pop('model_table', "model")
        train_query = train_func(**dict(config, **{"target": self.target_column}))
        _query_path = self.query_dir / f"{func_name}.sql"
        self.save_query(_query_path, train_query)

        return od({
            "td>": str(_query_path),
            "source": train_table,
            "create_table": model_table,
        })

    def _build_downsampling_task(
            self,
            source: str,
            target_column: str) -> OrderedDict:

        downsample_query = downsampling_rate("${source}", "${target_column}")
        _query_path = self.query_dir / "downsampling_rate.sql"
        self.save_query(_query_path, downsample_query)

        return od({
            "td>": str(_query_path),
            "source": source,
            "target_column": target_column,
            "engine": "presto",
            "store_last_results": True
        })

    def _build_predict_and_eval_task(
            self,
            config: OrderedDict,
            mod: object,
            pred_idx: int,
            test_table: str,
            metrics: List[str],
            multiple_predictors: bool = False) -> OrderedDict:

        func_name = config.pop('name')
        pred_func = getattr(mod, func_name)

        default_table = "prediction" if multiple_predictors else f"prediction_{pred_idx}"
        predict_table = config.pop("output_table", default_table)
        test_table = config.pop("target_table", test_table)
        model_table = config.pop("model_table", "model")

        predict_query, predicted_col = pred_func(**dict(config, **{"id_column": self.id_column}))
        _query_path = self.query_dir / f"{func_name}.sql"
        self.save_query(_query_path, predict_query)

        acc_template = "{metric}: ${{td.last_results.{metric}}}"
        acc_str = "\t".join(acc_template.format_map({"metric": metric}) for metric in metrics)

        return od({
            "+exec_predict": od({
                "td>": str(_query_path),
                "target_table": test_table,
                "create_table": predict_table,
                "model_table": model_table
            }),
            "+evaluate": od({
                "td>": str(self.query_dir / "evaluate.sql"),
                "actual": test_table,
                "predicted_table": predict_table,
                "predicted_column": predicted_col,
                "store_last_results": True
            }),
            "+show_accuracy": {"echo>": acc_str}
        })

    def dump_pipeline(
            self,
            config_file: str,
            dest_file: str = None,
            overwrite: bool = False) -> None:
        """Dump an ML pipeline, Hivemall SQLs and digdag workflows, from config yaml.

        Parameters
        ----------
        config_file : :obj:`str`
            Config file path.
        dest_file : :obj:`str`
            Destination of digdag workflow. Query directory should be written in a config file.
        overwrite : bool
            Flag whether overwrite output files or not. This option will remove query directory.

        Returns
        -------
        None

        """
        with open(config_file, "r") as f:
            config = yaml.load(f)

        source = config['source']
        dbname = config['dbname']

        self.id_column = config['id_column']
        self.target_column = config['target_column']
        train_sample_rate = config["train_sample_rate"]
        oversample_pos_n_times = config.get("oversample_pos_n_times")
        oversample_n_times = config.get("oversample_n_times")

        # Extract column related information.
        self._set_columns(config)

        workflow = od()  # type: OrderedDict[str, Any]
        export = od()  # type: OrderedDict[str, Any]
        # Since digdag "!include" seems to be a custom YAML tag, and can't find a way to dump with PyYAML...
        # export["!include"] = "config/params.yml"
        export["source"] = source
        export["train_sample_rate"] = train_sample_rate

        if oversample_n_times:
            export["oversample_n_times"] = oversample_n_times

        elif oversample_pos_n_times:
            export["oversample_pos_n_times"] = oversample_pos_n_times

        export["td"] = {"database": dbname, "engine": "hive"}
        workflow["_export"] = export

        self.query_dir = Path(config.get("query_dir", "queries"))

        if overwrite:
            shutil.rmtree(self.query_dir, ignore_errors=True)

        preparation = self._build_shuffle_and_split_task(stratify=config.get("stratify"))

        do_imputation = len(self.imputation_clauses) > 0
        do_normalization = len(self.normalization_clauses) > 0

        vectorize_target_train = f"{source}_train"
        vectorize_target_test = f"{source}_test"
        vectorize_target_whole = f"{source}_shuffled"

        if do_imputation or do_normalization:
            stats_query = compute_stats("${source}", self.numerical_columns)
            combined_stats = combine_train_test_stats("${source}", self.numerical_columns)
            stats_path = self.query_dir / "stats.sql"
            self.combine_stats_path = self.query_dir / "combine_stats.sql"
            self.save_query(stats_path, stats_query)
            self.save_query(self.combine_stats_path, combined_stats)

            self.comp_stats_task = od({
                "td>": str(stats_path),
                "engine": "presto",
                "source": source,
                "create_table": "${source}_stats"})

        if do_imputation:
            output_prefix = f"{source}_imputed"
            preparation["+imputation"], vectorize_target_train, vectorize_target_test = self._build_task_with_stats(
                query_basename="impute", source=source, source_whole=vectorize_target_whole,
                output_prefix=output_prefix,
                target_columns=self.imputed_columns, target_clauses=self.imputation_clauses,
                target_clauses_whole=self.imputation_clauses_whole)
            vectorize_target_whole = output_prefix

        if do_normalization:
            output_prefix = f"{source}_norm"
            preparation["+normalization"], vectorize_target_train, vectorize_target_test = self._build_task_with_stats(
                query_basename="normalize", source=f"{source}_imputed", source_whole=vectorize_target_whole,
                output_prefix=output_prefix,
                target_columns=self.normalized_columns, target_clauses=self.normalization_clauses,
                target_clauses_whole=self.normalization_clauses_whole, hive=True)
            vectorize_target_whole = output_prefix

        workflow["+preparation"] = preparation

        workflow["+vectorization"], train_table, test_table = self._build_vectorize_task(
            config.get("vectorizer", {}), source=vectorize_target_whole,
            source_train=vectorize_target_train, source_test=vectorize_target_test)

        # Preparation for loading train/predict functions dynamically
        __import__('molehill.model')
        mod = sys.modules['molehill.model']

        main = od()  # type: OrderedDict[str, Any]

        train_idx = 0
        train_tasks = od()  # type: OrderedDict[str, Any]
        for trainer in config.get('trainer', {}):
            if oversample_n_times and trainer.get('oversample_n_times') is None:
                trainer['oversample_n_times'] = "${oversample_n_times}"
            elif oversample_pos_n_times and trainer.get('oversample_pos_n_times') is None:
                trainer['oversample_pos_n_times'] = "${oversample_pos_n_times}"

            train_tasks[f"+train_{train_idx}"] = self._build_train_task(trainer, mod, train_table)
            train_idx += 1

        if train_idx > 1:
            train_tasks["_parallel"] = True

        main["+train"] = train_tasks

        # Save evaluation query before prediction
        metrics = config['evaluator']['metrics']
        evaluate_query = evaluate(metrics,
                                  target_column=self.target_column,
                                  target_table="${actual}",
                                  prediction_table="${predicted_table}",
                                  predicted_column="${predicted_column}")
        self.save_query(self.query_dir / "evaluate.sql", evaluate_query)

        pred_idx = 0
        pred_tasks = od()  # type: OrderedDict[str, Any]

        predictors = config.get('predictor', {})

        if oversample_pos_n_times:
            pred_tasks[f"+compute_downsampling_rate"] = self._build_downsampling_task(
                train_table, self.target_column)

        for predictor in predictors:
            if oversample_pos_n_times and predictor.get('oversample_pos_n_times') is None:
                predictor['oversample_pos_n_times'] = "${oversample_pos_n_times}"

            pred_tasks[f"+seq_{pred_idx}"] = self._build_predict_and_eval_task(
                predictor, mod, pred_idx, test_table, metrics, len(predictors) == 1)

            pred_idx += 1

        if pred_idx > 1:
            pred_tasks["_parallel"] = True

        main["+predict"] = pred_tasks
        workflow["+main"] = main

        self.workflow_path = dest_file if dest_file else f"{source}.dig"

        if not overwrite and Path(self.workflow_path).exists():
            raise FileExistsError(f"{self.workflow_path} already exists")

        self.save_query(self.workflow_path, yaml.dump(workflow, default_flow_style=False))
