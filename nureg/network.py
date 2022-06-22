"""
A collection of networks for the neutrino regression package
"""

from pathlib import Path
from typing import Union
import torch as T

import torch.nn as nn
from nflows import flows, distributions

from .torch_utils import count_parameters, sel_device
from .transforms import stacked_norm_flow
from .utils import save_yaml_files
from .modules import DeepSet, DenseNetwork


class MyNetBase(nn.Module):
    """A base class which is used to keep consistancy and harmony between the networks defined here
    and the trainer class
    """

    def __init__(
        self,
        *,
        name: str,
        save_dir: str,
        inpt_dim: Union[int, list],
        outp_dim: Union[int, list],
        device: str = "cpu",
        mkdir: bool = True,
    ) -> None:
        """
        kwargs:
            name: The name for the network, used for saving
            save_dir: The save directory for the model
            inpt_dim: The dimension of the input data
            outp_dim: The dimension of the output data
            device: The name of the device on which to load/save and store the network
            mkdir: If a directory for holding the model should be made
        """
        super().__init__()
        print(f"\nCreating network: {name}")

        ## Basic interfacing class attributes
        self.name = name
        self.save_dir = save_dir
        self.full_name = Path(save_dir, name)
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.device = sel_device(device)

        ## A list of all the loss names, all classes need a total loss!
        self.loss_names = ["total"]

        ## Create the folder to store the network
        if mkdir:
            self.full_name.mkdir(parents=True, exist_ok=True)

    def loss_dict_reset(self) -> dict:
        """Reset the loss dictionary
        - Returns a dictionary with 0 values for each of the loss names
        - Should be called at the beggining of each get_losses call
        """
        return {
            lsnm: T.tensor(0, dtype=T.float32, device=self.device)
            for lsnm in self.loss_names
        }

    def set_preproc(self, stat_dict):
        """Save a dictionary of data processing tensors as buffers on the network
        - Ensures they will be saved/loaded alongside the network
        """
        for key, val in stat_dict.items():
            self.register_buffer(key, val.to(self.device))

    def get_losses(self, _batch: tuple, _batch_idx: int) -> dict:
        """The function called by the trainer class to perform gradient descent
        by defualt the forward pass should have space for the sample and a get_loss
        flag

        Must return a dictionary of losses, and the descent will be performed on 'total'
        """
        loss_dict = {"total": 0}
        return loss_dict

    def visualise(self, *_args, **__kwargs):
        """This method should be overwritten by any inheriting network
        - It is used to save certain plots using a batch of samples
        """
        print("This model has no visualise method")

    def save(
        self,
        file_name: str = "model",
        as_dict: bool = False,
        cust_path: Union[str, Path] = "",
    ) -> None:
        """Save a version of the model
        - Will place the model in its save_dir/name/ by default
        - Can be saved as either as fixed or as a dictionary

        kwargs:
            name: The output name of the network file
            as_dict: True if the network is to be saved as a torch dict
            cust_path: The path to save the network, if empty uses the save_dir
        """

        ## All dict saved get the dict suffix
        if as_dict:
            file_name += "_dict"

        ## Check that the folder exists
        folder = Path(cust_path or self.full_name)
        folder.mkdir(parents=True, exist_ok=True)

        ## Create the full path of the file
        full_path = Path(folder, file_name)

        ## Use the torch save method
        if as_dict:
            T.save(self.state_dict(), full_path)
        else:
            T.save(self, full_path)

    def save_configs(self, data_conf, net_conf, train_conf):
        """Save the three config files that were used to build the network,
        supply the data and train the model
        """
        save_yaml_files(
            self.full_name / "config",
            ["data", "net", "train"],
            [data_conf, net_conf, train_conf],
        )

    def set_device(self, device):
        """Sets the device attribute and moves all parameters"""
        self.device = sel_device(device)
        self.to(self.device)

    def __repr__(self):
        return super().__repr__() + "\nNum params: " + str(count_parameters(self))


class SingleNeutrinoNSF(MyNetBase):
    """A conditional autoregressive neural spline flow for single neutrino regression
    - As this is an INN there is some variable name changes that need to be claified
        - The 'input' is usually the observed information, so here it becomes
          the contextual information
        - The 'output' or 'target' is the truth level information so here it becomes
          the features of the flow
    """

    def __init__(
        self,
        base_kwargs: dict,
        flow_kwargs: dict,
        dpset_kwargs: dict,
        embd_kwargs: dict,
    ) -> None:
        """
        base_kwargs will contain the input dimension expressed as a list for the
        [MET, leptons, misc, jets] attributes

        args:
            dpset_kwargs: A dictionary containing the keyword arguments for the deep set
            flow_kwargs: A dictionary containing the keyword arguments for the flow
            embd_kwags: Keyword arguments for the context embedding network
        """
        super().__init__(**base_kwargs)

        ## Initialise the deep set
        self.jet_ds = DeepSet(
            self.inpt_dim[-1],
            ctxt_dim=sum(self.inpt_dim[0:3]),  ## met, lepton, and misc inputs
            **dpset_kwargs,
        )

        ## Initialise the context embedding network
        self.embd_net = DenseNetwork(
            inpt_dim=sum(self.inpt_dim[0:3]) + self.jet_ds.outp_dim, **embd_kwargs
        )

        ## Save the flow: a combination of the inn and a gaussian
        inn = stacked_norm_flow(
            self.outp_dim, ctxt_dim=self.embd_net.outp_dim, **flow_kwargs
        )
        self.flow = flows.Flow(inn, distributions.StandardNormal([self.outp_dim]))

        ## Move the network to the selected device
        self.to(self.device)

    def get_flow_tensors(self, sample):
        """Given a sample (including the target), get the contextual and target tensors
        for the the normalising flow
        - called for forward passes
        - includes passing the jets through the deep set
        """

        ## Unpack the sample
        met, lept, misc, jets, neut = sample

        ## Pass the data through a conditional deep set
        jets = self.jet_ds.forward(
            tensor=jets,
            mask=jets[..., 0] != -T.inf,
            ctxt=T.cat([met, lept, misc], dim=-1),
        )
        ctxt = T.cat([met, lept, misc, jets], dim=-1)

        ## Pass through the embedding network
        ctxt = self.embd_net(ctxt)

        ## Return the context AND the target
        return ctxt, neut

    def get_losses(self, sample: tuple, _batch_idx: int) -> dict:
        """For this model there is only one loss and that is the log probability"""
        ctxt, neut = self.get_flow_tensors(sample)
        loss_dict = self.loss_dict_reset()
        loss_dict["total"] = -self.flow.log_prob(neut, context=ctxt).mean()
        return loss_dict

    def generate(self, sample, n_points=1) -> T.Tensor:
        """Generate points in the X space by sampling from the latent"""
        ctxt, _ = self.get_flow_tensors(sample)
        generated = self.flow.sample(n_points, context=ctxt).squeeze()
        return generated

    def sample_and_log_prob(self, sample, n_points=100) -> T.Tensor:
        """Generate many points per sample and return all with their log likelihoods"""
        ctxt, _ = self.get_flow_tensors(sample)
        return self.flow.sample_and_log_prob(n_points, ctxt)

    def get_mode(self, sample, n_points=100) -> T.Tensor:
        """Generate points, then select the one with the most likely value in the
        reconstruction space
        """
        gen, log_like = self.sample_and_log_prob(sample, n_points=n_points)
        return gen[T.arange(len(gen)), log_like.argmax(dim=-1)]

    def forward(self, sample):
        """Get the latent space embeddings given neutrino and context information"""
        ctxt, neut = self.get_flow_tensors(sample)
        return self.flow.transform_to_noise(neut, context=ctxt)
