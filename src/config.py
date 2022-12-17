"""
Sets the whole configuration globally based on the conf/ directory.

Every possible constant, path, number should be located in the config (with sane defaults)
"""

import json
import hydra
from omegaconf import OmegaConf


hydra.initialize(version_base=None, config_path="../conf", job_name="app")
_hydra_dict_config = hydra.compose(config_name="config")
OmegaConf.resolve(
    _hydra_dict_config
)  # some values in the config might be express via other values in the config. resolve evaluates the config and saves only concrete values
OmegaConf.set_readonly(_hydra_dict_config, True)
OmegaConf.set_struct(
    _hydra_dict_config, True
)  #  don't allow setting arbitrary fields https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#struct-flag
cfg = OmegaConf.to_container(_hydra_dict_config)
# cfg object is a globally available config


def cfg_pretty_str() -> str:
    return json.dumps(cfg, indent=2)


if __name__ == "__main__":
    print(cfg_pretty_str())
