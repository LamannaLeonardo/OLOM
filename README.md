# Online Learning of Object-Centric Symbolic Models in Partially Observable Environments
<!-- Define badges -->
<div style="display: flex; gap: 10px;">
   
  <a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" height="20"/></a>

</div>


Code for the ICAART 2026 paper "Online Learning of Object-Centric Symbolic Models in Partially Observable Environments". 
L. Lamanna, L. Serafini, A. Saffiotti and P. Traverso

## Installation

1. Clone this repository:
```
git clone https://github.com/LamannaLeonardo/OLOM.git
```

1. Create a Python `3.10` environment; for example with conda:
```
cd OLOM && conda create -n olom python=3.10
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

## Citations
```
@article{lamanna2026online,
  title={Online Learning of Object-Centric Symbolic Models in Partially Observable Environments},
  author={Lamanna, Leonardo and Serafini, Luciano and Saffiotti, Alessandro and Traverso, Paolo},
  booktitle={Proceedings of the International Conference on Agents and Artificial Intelligence},
  volume={18},
  year={2026}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.
