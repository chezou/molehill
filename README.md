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
# will generate queries directory and titanic.dig
$ generate_workflow --overwrite -dest titanic.dig resources/titanic_pipeline.yml
$ td wf push proj-name
$ td wf start proj-name titanic --session now
```

## Examples

Example YAML files can be found as follows:

The following YAMLs are example config files for [Titanic](https://github.com/amueller/scipy-2017-sklearn/blob/master/notebooks/datasets/titanic3.csv) survival prediction with Logistic Regression.
  
- [titanic_pipeline.yml](./resources/titanic_pipeline.yml)
  - Example workflow for Linear Regression and Random Forest
  - Generated files are under [examples/titanic](./examples/titanic)
- [titanic_pipeline_rf.yml](resources/titanic_pipeline_rf.yml)
  - Example workflow for Random Forest
  - Generated files are under [examples/titanic_rf](./examples/titanic_rf)
- [titanic_pipeline_oversample.yml](resources/titanic_pipeline_oversample.yml)
  - Example workflow for Linear Regression with oversampling
  - Generated files are under [examples/titanic_oversample](./examples/titanic_oversample)
- [titanic_pipeline_pos_oversample.yml](resources/titanic_pipeline_pos_oversample.yml)
  - Example workflow for Linear Regression with oversampling for positive class
  - Generated files are under [examples/titanic_pos_oversample](./examples/titanic_pos_oversample)
