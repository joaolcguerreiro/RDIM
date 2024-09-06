# <span style="color: orange">Experiment Tracking</span>

#### Directory Structure

Everytime you train a model, an experiment folder is built, unless training is resumed, in which case the folder from the prior run is used.

Experiments folder contains logs and settings from prior runs, useful to resume training, evaluate checkpoints, track parameters, etc. For instance, the configuration file for the execution is stored, allowing the original parameters to be retrieved when resuming training.

The experiment folder encompasses configuration files, training checkpoints, logs, figures, and results. Its structure is the following:

```
├───experiments
      ├───example_experiment_01
      │      ├───checkpoints
      │      │      └──1.pth
      │      │      └──2.pth
      │      │      └──3.pth
      │      ...    └──...
      │      │
      │      ├───configs
      │      │      └──config.yaml
      │      │
      │      ├───figures
      │      │      ├───test
      │      │      └───train
      │      │
      │      ├───logs
      │      │      ├───test
      │      │      └───train
      │      │
      │      └───results
      │      │      └──example_result_0001.nii.gz
      │      │      └──example_result_0002.nii.gz
      │      │      └──example_result_0003.nii.gz
      │      ...    └──...
      │
      ├───example_experiment_02
      │      ├───checkpoints
      │      ├───configs
      │      ├───figures
      │      ├───logs
      │      └───results
      ...
```

<hr style="border:1px solid gray">

#### Folder Description

- `configs/` contains copy of configuration file used in the experiment. Loaded when resuming training.

---

- `checkpoints/` stores saved checkpoints of the model during training

---

- `figures/` contains visualizations for better analysis of the experiment.

---

- `logs/` contains logs from training and testing executions.

---

- `results/` stores the output of the trained model when evaluated on the test set.
