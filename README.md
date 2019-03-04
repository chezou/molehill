# molehill

[![CircleCI](https://circleci.com/gh/chezou/molehill.svg?style=svg)](https://circleci.com/gh/chezou/molehill)

Combine Apache Hivemall(incubating) and digdag together.

Generate Hivemall queries and Digdag workflow for TreasureData from YAML file.

## Installation

```bash
$ pip install git+https://github.com/chezou/molehill#egg=molehill
```

## Usage

```bash
$ generate_workflow --overwrite -dest titanic.rf resources/titanic_pipeline.yml
$ td wf push proj-name
$ td wf start proj-name titanic --session now
```

## Examples

Example YAML files can be found as follows:

- [titanic_pipeline.yml](./resources/titanic_pipeline.yml)
  - Example config file for [Titanic](https://github.com/amueller/scipy-2017-sklearn/blob/master/notebooks/datasets/titanic3.csv) survival prediction with Logistic Regression.
  - Generated files are under [examples/titanic](./examples/titanic)
- [titanic_rf_pipeline.yml](./resources/titanic_rf_pipeline.yml)
  - Example config file for [Titanic](https://github.com/amueller/scipy-2017-sklearn/blob/master/notebooks/datasets/titanic3.csv) survival prediction with Random Forest.
  - Generated files are under [examples/titanic_rf](./examples/titanic_rf)
