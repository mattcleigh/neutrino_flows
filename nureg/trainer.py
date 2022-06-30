"""
Base class for training network
"""

import json
from pathlib import Path
from functools import partialmethod

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .plotting import plot_multi_loss
from .torch_utils import (
    RunningAverage,
    get_optim,
    get_sched,
    move_dev,
    get_grad_norm,
)


class Trainer:
    """A class to oversee the training of a network which can handle its own losses"""

    def __init__(
        self,
        network: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader = None,
        patience: int = 100,
        max_epochs: int = 100,
        grad_clip: float = 0.0,
        optim_dict: dict = None,
        sched_dict: dict = None,
        vis_every: int = 10,
        chkp_every: int = 10,
        resume: bool = False,
        tqdm_quiet: bool = False,
        quick_mode: int = 0,
    ) -> None:
        """
        args:
            network:      Network with a get_losses method
            train_loader: Dataloader on which to perform batched gradient descent
        kwargs:
            valid_loader: Dataloader for validation loss and early stopping
            patience:     Early stopping patience calculated using the validation set
            max_epochs:   Maximum number of epochs to train for
            grad_clip:    Clip value for the norm of the gradients (0 will not clip)
            n_workers:    Number of parallel threads which prepare each batch
            optim_dict:   A dict used to select and configure the optimiser
            sched_dict:   A dict used to select and configure the scheduler
            vis_every:    Run the network's visualisation function every X epochs
            chkp_every:   Save a checkpoint of the net/opt/schd/loss every X epochs
            resume:       Load the 'latest' checkpoint of the trainer object
            tqdm_quiet:   Prevents tqdm loading bars, good for gridjobs writing to logs
            quick_mode:   Break the training epoch after X many batches, for debugging
        """
        print("\nInitialising the trainer")

        ## Default dict arguments
        self.optim_dict = optim_dict.copy() or {"name": "adam", "lr": 1e-4}
        self.sched_dict = sched_dict.copy() or {"name": "none"}

        ## Save the network and dataloaders
        self.network = network
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.has_v = valid_loader is not None

        ## Its also really useful to save the underlying dataset objects (by reference)
        self.train_set = train_loader.dataset
        if isinstance(self.train_set, Subset):
            self.train_set = self.train_set.dataset
        if self.has_v:
            self.valid_set = valid_loader.dataset
            if isinstance(self.valid_set, Subset):
                self.valid_set = self.valid_set.dataset

        ## Report on the number of files/samples used (keep as the subset for length!)
        print(f"train set: {len(self.train_loader.dataset):7} samples")
        if self.has_v:
            print(f"valid set: {len(self.valid_loader.dataset):7} samples")
        else:
            print("No validation set provided")

        ## Create a history of train and validation losses for early stopping
        self.loss_hist = {
            lsnm: {dset: [] for dset in ["train", "valid"]}
            for lsnm in self.network.loss_names
        }

        ## A running average tracker for each loss during an epoch
        self.run_loss = {
            lsnm: RunningAverage(dev=self.network.device)
            for lsnm in self.network.loss_names
        }

        ## Gradient clipping settings and saving settings
        self.grad_clip = grad_clip
        self.vis_every = vis_every
        self.chkp_every = chkp_every

        ## Load the optimiser and scheduler
        self.optimiser = get_optim(self.optim_dict, self.network.parameters())
        self.scheduler = get_sched(
            self.sched_dict,
            self.optimiser,
            len(self.train_loader),
        )

        ## Variables to keep track of stopping conditions
        self.max_epochs = max_epochs
        self.patience = patience
        self.num_epochs = 0
        self.bad_epochs = 0
        self.best_epoch = 0

        ## For quick_mode operations (one batch pass per epoch)
        self.quick_mode = quick_mode
        if self.quick_mode:
            print(" - quickmode activated (should be only for debugging!)")

        ## Turning off tqdm (for sbatch logfiles)
        if tqdm_quiet:
            print(" - disabling tqdm outputs")
            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        ## Load the previous checkpoint
        if resume:
            self.load_checkpoint(flag="latest")
            self._save_chpt_dict("before_resume")  ## Make a save before continuing

    def explode_learning(
        self,
        init_lr: float = 1e-5,
        finl_lr: float = 1,
        n_iter: int = 200,
        scheme: str = "exp",
    ):
        """This method slowly increases the learning rate so one can find the max
        stable value to use for training.
        - create output plots of loss recorded per batch as lr increases
        - should not train immediately after!

        kwargs:
            init_lr: The initial learning rate at the start of the test
            stop: The final learning rate at the end of the test (might not be reached)
            n_iter: The number of batch passes to test
            scheme: The lr increase scheme, either 'lin' or 'exp'
        """
        print(f"\nExploding the learning rate from {init_lr} to {finl_lr}")

        ## Turn off the scheduler and turn on quick mode silence tqdm
        self.scheduler = None
        self.quick_mode = 1
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        ## Create the list of learning rates to try
        ## Calculate the new learning rate
        if scheme == "lin":
            lrs = [(finl_lr - init_lr) / n_iter * e + init_lr for e in range(n_iter)]
        elif scheme == "exp":
            lrs = [
                init_lr * np.power(finl_lr / init_lr, e / n_iter) for e in range(n_iter)
            ]
        else:
            raise ValueError(f"Unrecognised lr increase sheme: {scheme}")

        ## Cycle through each learning rate
        grad_norms = []
        for i, lr in enumerate(lrs):
            print(i / n_iter, end="\r")

            ## Set the optimiser learning rate
            for g in self.optimiser.param_groups:
                g["lr"] = lr

            ## Do one batch
            self.epoch(is_train=True)

            ## Store the norm of the gradients
            grad_norms.append(get_grad_norm(self.network))

        ## Create the folder for the plots
        test_folder = Path(self.network.full_name, "lr_test")
        test_folder.mkdir(parents=True, exist_ok=True)

        ## Plot the gradient norm
        plt.plot(grad_norms, label="gradient norm")
        plt.yscale("log")
        plt.legend()
        plt.savefig(Path(test_folder, "gnorm_vs_epoch.png"))
        plt.close()

        ## Plot the lr vs epoch
        plt.plot(lrs, label="learning rate")
        plt.legend()
        plt.savefig(Path(test_folder, "lr_vs_epoch.png"))
        plt.close()

        ## Plot the loss vs lr
        plot_multi_loss(
            Path(test_folder, "loss_vs_lr.png"),
            self.loss_hist,
            xvals=lrs,
            xlabel="learning rate",
            logx=True,
        )

    def run_training_loop(self) -> None:
        """The main loop which cycles epochs of train and test
        - After each epochs it calls the save function and checks for early stopping
        """
        print("\nStarting the training process")

        for epc in np.arange(self.num_epochs, self.max_epochs):
            print(f"\nEpoch: {epc}")

            ## Run the test/train cycle, update stats, and save
            self.epoch(is_train=True)
            if self.has_v:
                self.epoch(is_train=False)
            self.count_epochs()
            self.save_checkpoint()

            ## Check if we have exceeded the patience
            if self.bad_epochs > self.patience:
                print(" - patience Exceeded: Stopping training!")
                return 0

        ## If we have reached the maximum number of epochs
        print(" - maximum number of epochs exceeded")
        return 0

    def epoch(self, is_train: bool = False) -> None:
        """Perform a single epoch on either the train loader or the validation loader
        - Will update average loss during epoch
        - Will add average loss to loss history at end of epoch

        kwargs:
            is_train: Effects gradient tracking, network state, and data loader
        """

        ## Select the correct mode for the epoch
        if is_train:
            mode = "train"
            self.network.train()
            loader = self.train_loader
            T.set_grad_enabled(True)
        else:
            mode = "valid"
            self.network.eval()
            loader = self.valid_loader
            T.set_grad_enabled(False)

        ## Cycle through the batches provided by the selected loader
        for batch_idx, batch in enumerate(tqdm(loader, desc=mode, ncols=80)):

            ## Move the sample to the network device
            batch = move_dev(batch, self.network.device)

            ## Pass through the network and get the loss dictionary
            losses = self.network.get_losses(batch, batch_idx)

            ## For training epochs we perform gradient descent
            if is_train:

                ## Zero and calculate gradients using total loss (from dict)
                self.optimiser.zero_grad(set_to_none=True)
                losses["total"].backward()

                ## Apply gradient clipping
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)

                ## Step the optimiser
                self.optimiser.step()

                ## Step the learning rate scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

            ## Update the each of the running losses using the dictionary
            for lnm, running in self.run_loss.items():
                running.update(losses[lnm].detach())

            ## Break when using quick mode
            if self.quick_mode > 0 and batch_idx >= self.quick_mode:
                break

        ## Use the running losses to update the total history, then reset
        for lnm, running in self.run_loss.items():

            ## Update my own dict and reset
            self.loss_hist[lnm][mode].append(running.avg)
            running.reset()

    def count_epochs(self) -> None:
        """Update attributes counting number of bad and total epochs"""
        self.num_epochs = len(self.loss_hist["total"]["train"])
        if self.has_v:
            self.best_epoch = np.argmin(self.loss_hist["total"]["valid"]) + 1
            self.bad_epochs = self.num_epochs - self.best_epoch

    def _save_chpt_dict(self, sv_name: str) -> None:
        """Create a saved dict of the network/optimiser/loss/scheduler states for
        resuming training at a later stage
        """

        ## Create checkpoint folder
        chckpnt_folder = Path(self.network.full_name, "checkpoints")
        chckpnt_folder.mkdir(parents=True, exist_ok=True)

        ## Save a checkpoint of the network/optimiser/loss (for reloading)
        checkpoint = {
            "network": self.network.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "losses": self.loss_hist,
        }

        ## Add the scheduler if used in training
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        ## Save using pytorch's pickle method and the save name
        T.save(checkpoint, Path(chckpnt_folder, f"checkpoint_{sv_name}"))

    def save_checkpoint(self) -> None:
        """Add folders to the network's save directory containing
        - losses -> Loss history as json and plotted as pngs
        - checkpoints -> Checkpoints of network/optimiser/scheduler/loss states
        - visual -> Output of network's visualisation method on first valid batch
        """

        print(" - saving...")

        ## Save best network model and dict (easier to reload) in main directory
        if self.bad_epochs == 0:
            self.network.save("best")
            # self.network.save("best", as_dict=True) ## Takes too long for large models

        ## Create loss folder
        loss_folder = Path(self.network.full_name, "losses")
        loss_folder.mkdir(parents=True, exist_ok=True)

        ## Save a plot and a json of the loss history
        loss_file_name = Path(loss_folder, "losses.json")
        plot_multi_loss(loss_file_name, self.loss_hist)
        with open(loss_file_name, "w", encoding="utf-8") as l_file:
            json.dump(self.loss_hist, l_file, indent=2)

        ## Always save latest version of checkpoint
        self._save_chpt_dict("latest")

        ## For checkpointing at regular intervals
        if self.chkp_every > 0 and self.num_epochs % self.chkp_every == 0:
            self._save_chpt_dict(self.num_epochs)

        ## For visualisation
        if self.vis_every > 0 and self.num_epochs % self.vis_every == 0:

            ## Set evaluation mode
            self.network.eval()
            T.set_grad_enabled(False)

            ## Create the vis folder
            vis_folder = Path(self.network.full_name, "visual")
            vis_folder.mkdir(parents=True, exist_ok=True)

            ## The most general way I have found is to pass the dataloader itself!
            ## This does not require much memory (passed by reference) and it means that
            ## visualise can run over the whole dataset (performance metrics)
            ## while still having access to labels, norm states and other dset features
            self.network.visualise(
                self.valid_loader if self.has_v else self.train_loader,
                path=vis_folder,
                flag=str(self.num_epochs),
            )

    def load_checkpoint(self, flag="latest") -> None:
        """Loads the latest instance of a saved network to continue training"""
        print(" - loading checkpoint...")

        ## Load the and unpack checkpoint object
        checkpoint = T.load(
            Path(self.network.full_name, "checkpoints", f"checkpoint_{flag}")
        )
        self.network.load_state_dict(checkpoint["network"])
        self.optimiser.load_state_dict(checkpoint["optimiser"])
        self.loss_hist = checkpoint["losses"]
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            checkpoint["scheduler"] = self.scheduler.state_dict()

        ## Update the epoch count
        self.count_epochs()

        ## Cant resume if trained enough, parameters should be changed
        if self.bad_epochs > self.patience:
            raise ValueError("Loaded checkpoint already exeeds specified patience!")
        if self.num_epochs > self.max_epochs:
            raise ValueError("Loaded checkpoint already trained up to max epochs!")
