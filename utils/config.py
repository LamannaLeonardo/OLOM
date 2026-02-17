from contextvars import ContextVar

import yaml

ag_cfg = ContextVar("ag_cfg")
dom_cfg = ContextVar("dom_cfg")
run_cfg = ContextVar("run_cfg")


def load_ag_cfg(cfg_file):
    with open(cfg_file) as f:
        config = yaml.safe_load(f)
    ag_cfg.set(config)


def load_dom_cfg(cfg_file):
    with open(cfg_file) as f:
        dom_cfg.set(yaml.safe_load(f))


def load_run_cfg(cfg_file):
    with open(cfg_file) as f:
        run_cfg.set(yaml.safe_load(f))


def get_ag_cfg():
    return ag_cfg.get(None)


def get_dom_cfg():
    return dom_cfg.get(None)


def get_run_cfg():
    return run_cfg.get(None)
