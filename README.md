# molehill

Combine Apache Hivemall(incubating) and digdag together.

Generate Hivemall queries and Digdag workflow for TreasureData from YAML file.

## Usage

```bash
$ pip install -r requirements.txt -c constraints.txt
$ python generate_workflow.py --overwrite -dest titanic.rf resources/titanic_pipeline.yml
$ td wf push proj-name
$ td wf start proj-name titanic --session now
```

## Examples

See:

- [titanic_pipeline.yml](./resources/titanic_pipeline.yml)
  - Example config file for [Titanic](https://github.com/amueller/scipy-2017-sklearn/blob/master/notebooks/datasets/titanic3.csv) survival prediction with Logistic Regression.
- [titanic_rf_pipeline.yml](./resources/titanic_rf_pipeline.yml)
  - Example config file for [Titanic](https://github.com/amueller/scipy-2017-sklearn/blob/master/notebooks/datasets/titanic3.csv) survival prediction with Random Forest.
