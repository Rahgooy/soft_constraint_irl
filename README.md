# Max Entropy Soft Constraint Inverse Reinforcement Learning

You can find some examples from the grid-world in the notebooks folder.

## Installing the requirements

```bash
  pip install -r requirements.txt
```

## Running the experiments

### Hard constraints

To learn the constraints run:

```bash
  python -m max_ent.examples.learn_hard_constraints
```

After learning, run the following to generate the reports (in `./reports/hard/` folder):

```bash
  python -m max_ent.examples.compare_hard_results
```

### Soft constraints

To learn the constraints run:

```bash
  python -m max_ent.examples.learn_soft_constraints
```

After learning, run the following to generate the reports (in `./reports/soft/` folder):

```bash
  python -m max_ent.examples.compare_soft_results
```

### Orchestration

Run the notebook in `./notebooks/new_metrics.ipynb` .

Also, you can set `learn = True` in `./max_ent/examples/orchestrator_exp.py` then run:

```bash
  python -m max_ent.examples.orchestrator_exp
```

After that, set `learn = False` and run the above command again.
The reports will be generated into `./reports/orchestrator/` folder.

## Acknowledgement

This repository uses and modifies some codes from [irl-maxent](https://github.com/qzed/irl-maxent) library.
