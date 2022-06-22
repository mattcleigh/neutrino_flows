"""
Main callable script to train the neutrino regressor
"""

import torch as T
from torch.utils.data import DataLoader

from nureg.trainer import Trainer
from nureg.network import SingleNeutrinoNSF
from nureg.datasets import NuRegData
from nureg.utils import save_yaml_files, load_yaml_files
from nureg.torch_utils import train_valid_split

## Manual seed for reproducibility
T.manual_seed(42)


def main():
    """Run the script"""

    ## Load each of the dictionaries from the config files
    data_conf, flow_conf, train_conf = load_yaml_files(
        ["config/data.yaml", "config/flow.yaml", "config/train.yaml"]
    )

    ## Load the ttbar regression data
    train_set = NuRegData(dset="train", **data_conf)

    ## Get the data dimensions and add them the network configuraiton
    all_dims = train_set.get_dim()
    flow_conf["base_kwargs"]["outp_dim"] = all_dims.pop(-1)  ## Neutrino dim is last
    flow_conf["base_kwargs"]["inpt_dim"] = all_dims

    ## Initialise the network, creating a save directory in the process
    network = SingleNeutrinoNSF(**flow_conf)

    ## Store the configuration dicts in the network's save directory
    save_yaml_files(
        network.full_name / "config",
        ["data", "net", "train"],
        [data_conf, flow_conf, train_conf],
    )

    ## Fit the preprocessing scalers and make plots before and after processing
    train_set.plot_variables(network.full_name / "train_dist")
    scalers = train_set.save_preprocess_info(network.full_name / "scalers")
    train_set.apply_preprocess(scalers)
    train_set.plot_variables(network.full_name / "train_dist")

    ## Create the validation and training datasets
    train_set, valid_set = train_valid_split(
        train_set, train_conf["val_frac"], rand_split=False
    )

    ## Load the trainer
    trainer = Trainer(
        network,
        train_loader=DataLoader(train_set, **train_conf["loader_kwargs"]),
        valid_loader=DataLoader(valid_set, **train_conf["loader_kwargs"]),
        **train_conf["trainer_kwargs"]
    )

    ## Run the training looop
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
