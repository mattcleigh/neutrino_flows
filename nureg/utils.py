"""
Miscellaneous utility functions for the nureg package
"""

from pathlib import Path
from typing import Union
from copy import deepcopy
import yaml

import numpy as np
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from .torch_utils import get_act


def change_kwargs_for_made(old_kwargs):
    """Converts a dictionary of keyword arguments for configuring a
    DenseNetwork to one that can initialise a MADE network for the nflows package
    with similar (not exact) hyperparameters
    """
    new_kwargs = deepcopy(old_kwargs)

    ## Certain keys must be changed
    key_change(new_kwargs, "ctxt_dim", "context_features")
    key_change(new_kwargs, "drp", "dropout_probability")
    key_change(new_kwargs, "do_res", "use_residual_blocks")

    ## Certain keys are changed and their values modified
    if "act_h" in new_kwargs:
        new_kwargs["activation"] = get_act(new_kwargs.pop("act_h"))
    if "nrm" in new_kwargs:  ## Only has batch norm!
        new_kwargs["use_batch_norm"] = new_kwargs.pop("nrm") is not None

    ## Some options are missing
    missing = ["ctxt_in_all", "n_lyr_pbk", "act_o", "do_out"]
    for miss in missing:
        if miss in new_kwargs:
            del new_kwargs[miss]

    ## The hidden dimension passed to MADE as an arg, not a kwarg
    if "hddn_dim" in new_kwargs:
        hddn_dim = new_kwargs.pop("hddn_dim")
    else:
        ## Use the same default value for .modules.DenseNet
        hddn_dim = 32

    return new_kwargs, hddn_dim


def save_yaml_files(
    path: str, file_names: Union[str, list, tuple], dicts: Union[dict, list, tuple]
) -> None:
    """Saves a collection of yaml files in a folder
    - Makes the folder if it does not exist
    """

    ## If the input is not a list then one file is saved
    if isinstance(file_names, (str, Path)):
        with open(f"{path}/{file_names}.yaml", "w", encoding="UTF-8") as f:
            yaml.dump(dicts, f, sort_keys=False)
        return

    ## Make the folder
    Path(path).mkdir(parents=True, exist_ok=True)

    ## Save each file using yaml
    for f_nm, dic in zip(file_names, dicts):
        with open(f"{path}/{f_nm}.yaml", "w", encoding="UTF-8") as f:
            yaml.dump(dic, f, sort_keys=False)


def load_yaml_files(files: Union[list, tuple, str]) -> tuple:
    """Loads a list of files using yaml and returns a tuple of dictionaries"""

    ## If the input is not a list then it returns a dict
    if isinstance(files, (str, Path)):
        with open(files, encoding="utf-8") as f:
            return yaml.safe_load(f)

    opened = []

    ## Load each file using yaml
    for fnm in files:
        with open(fnm, encoding="utf-8") as f:
            opened.append(yaml.safe_load(f))

    return tuple(opened)


def get_scaler(name: str):
    """Return a sklearn scaler object given a name"""
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "power":
        return PowerTransformer()
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal")
    if name == "none":
        return None
    raise ValueError(f"No sklearn scaler with name: {name}")


def signed_angle_diff(angle1, angle2):
    """Calculate diff between two angles reduced to the interval of [-pi, pi]"""
    return (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi


def key_change(dic: dict, old_key: str, new_key: str, new_value=None) -> None:
    """Changes the key used in a dictionary inplace only if it exists"""

    ## If the original key is not present, nothing changes
    if old_key not in dic:
        return

    ## Use the old value and pop. Essentially a rename
    if new_value is None:
        dic[new_key] = dic.pop(old_key)

    ## Both a key change AND value change. Essentially a replacement
    else:
        dic[new_key] = new_value
        del dic[old_key]
