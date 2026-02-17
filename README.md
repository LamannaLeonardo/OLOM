

## Installation

1. Create a Python `3.10` environment; for example with conda:
```
conda create -n olom python=3.10
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run `main.py` script to reproduce the paper results for `OLOM`, different approaches 
and domains can be selected in `main.py`, through the following variables:
```
    train = True  # train the agent separately on every training environment
    test = False  # evaluate the agent separately on every test set of environments
    approaches = ['oracle']  # approach to be tested (e.g. rppo, drqn)
    domains = ['MNISTExib-v0', 'simpleMNISTExib-v0']  # simpleMNIST corresponds to DO in the paper
    nseeds = 5  # number of randomized runs to perform 
```

For every variant (e.g. `oracle`), the log results are stored in directory `res/`, e.g. `res/oracle`.

## Hyperparameter tuning
The code used for tuning hyperparameters of `RPPO` and `IMPALA` is available in `tune.py`; 
the one for `DRQN` is `drqn_tune.py`. The hyperparameter search space for every agent are defined in `configs/agents`.

## Datasets
The training/testing environments are detailed in directory `datasets` for
both domains `MNISTExib` (i.e. UO in the paper) and `simpleMNISTExib` (i.e. DO in the paper).
The subfolder `datasets/train` contains the environments used for hyperparameters tuning, the 
subfolder `datasets/fine-tune` the ones in the experimental evaluation.
The script used for generating all environments can be inspected in `utils/generators` and the
generation process is reproducible.
