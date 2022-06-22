# neutrino_flows
A public repository for a minimal working example in the neutrino flows project

This repository facilitates the steps required the produce and save a fully trained
conditional normalising flow using the pytorch and nflows libraries.

The main executable script is train.py and it performs the following steps:
- Loads in the three configuration dictionaries for the session (more on these below)
- Initialises the dataset for training using the data config
- Initialises the flow for training using the flow config and the dimensionality of samples in the dataset
- Creates a save directory for the flow into which it stores:
    - Copies of the configs
    - Histograms of the raw dataset features
    - Preprocessing scalers, which are fit on the dataset
    - Histograms of the pre-processed dataset features
    - Later this will also be filled with the loss values recorded during training, model checkpoints, and the "best" model based on lowest validation loss
- Splits the dataset into a training and a holdout vadiation set
- Initialises a Trainer using the train config
- Performs the learning through the Trainer's run_training_loop method

The Trainer class is just a simple class the performs subsequent iterations of gradient descent.
It is fully configured by the train.yaml file which allows it to facilitate things like optimiser configurations, checkpointing, early stopping etc.
Moving data onto and off of the network's training device is done automatically by the trainer.

The entrie session is configured by the tree yaml files which are found in config/
- data.yaml: Controls the data loading, where to file the files, which variables to use, which scalers to apply etc
- flow.yaml: Controls the configuration of the conditional INN, from the number of layers in the flow to the size of the deep set
- train.yaml: Controls the training set, here one can configure learning rate schedulers, gradient clipping, early stopping etc.

Each of these yaml files essentially define a hierarcy of keyword arguments which are passed to the functions in nureg/.
In each of these file there are extensive comments outlining what each option does.

To run this script:
1) Setup the environment
    - Either use the requirement.txt file to setup the appropriate python packages.
        - This project was tested with python 3.9
    - Alternatively use the docker build file to create an image which can run
    - Download the pre-built docker image from dockerhub here:
        - TODO add url
2) Download the data
    - The datafiles are a bit larger and thus are not stored in this repository
    - You can find them here:
        - TODO: add url
    - Make sure that the "path" keyword in config/train.yaml points to the downloaded folder
3) Specify the save path for the flow
    - This is done by setting the "base_kwargs/name" and "base_kwargs/save_dir" in config/flow.yaml
    - The code will try and create the directory if it does not exist
4) Run the scipt
    - Use simply "python train.py" and watch it go!

