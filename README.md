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
