"""
Collection of pytorch modules that make up the networks used in this package
"""

from typing import Union

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from .torch_utils import get_act, get_nrm, pass_with_mask, masked_pool, smart_cat


class MLPBlock(nn.Module):
    """A simple MLP block that makes up a dense network

    Made up of several layers containing:
    - linear map
    - activation function
    - layer normalisation
    - dropout

    Only the input of the block is concatentated with context information
    For residual blocks, the input is added to the output of the final layer
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        n_layers: int = 1,
        act: str = "lrlu",
        nrm: str = "none",
        drp: float = 0,
        do_res: bool = False,
    ) -> None:
        """
        args:
            inpt_dim: The number of features for the input layer
            outp_dim: The number of output features
        kwargs:
            ctxt_dim: The number of contextual features to concat to the inputs
            act: A string indicating the name of the activation function
            nrm: A string indicating the name of the normalisation
            drp: The dropout probability, 0 implies no dropout
            do_res: Add to previous output, only if dim does not change
            n_layers: The number of transform layers in this block
        """
        ## Applies the
        super().__init__()

        ## Save the input and output dimensions of the module
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim

        ## If this layer includes an additive residual connection
        self.do_res = do_res and (inpt_dim == outp_dim)

        ## Initialise the block layers as a module list
        self.block = nn.ModuleList()
        for n in range(n_layers):

            ## Increase the input dimension of the first layer to include context
            lyr_in = inpt_dim + ctxt_dim if n == 0 else outp_dim

            ## Linear transform, activation, normalisation, dropout
            self.block.append(nn.Linear(lyr_in, outp_dim))
            if act != "none":
                self.block.append(get_act(act))
            if nrm != "none":
                self.block.append(get_nrm(nrm, outp_dim))
            if drp > 0:
                self.block.append(nn.Dropout(drp))

    def forward(self, inpt: T.Tensor, ctxt: T.Tensor = None) -> T.Tensor:
        """
        args:
            tensor: Pytorch tensor to pass through the network
            ctxt: The conditioning tensor, can be ignored
        """

        ## Concatenate the context information to the input of the block
        if self.ctxt_dim and ctxt is None:
            raise ValueError(
                "Was expecting contextual information but none has been provided!"
            )
        temp = T.cat([inpt, ctxt], dim=-1) if self.ctxt_dim else inpt

        ## Pass through each transform in the block
        for layer in self.block:
            temp = layer(temp)

        ## Add the original inputs again for the residual connection
        if self.do_res:
            temp += inpt

        return temp

    def __repr__(self):
        string = str(self.inpt_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += "->"
        string += "->".join([str(b).split("(", 1)[0] for b in self.block])
        string += "->" + str(self.outp_dim)
        if self.do_res:
            string += "(add)"
        return string


class DenseNetwork(nn.Module):
    """A dense neural network made from a series of consecutive MLP blocks and context
    injection layers"""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int = 0,
        ctxt_dim: int = 0,
        hddn_dim: Union[int, list] = 32,
        num_blocks: int = 2,
        n_lyr_pbk: int = 1,
        act_h: str = "lrlu",
        act_o: str = "none",
        do_out: bool = True,
        nrm: str = "none",
        inpt_nrm: str = "none",
        drp: float = 0,
        do_res: bool = False,
        ctxt_in_all: bool = False,
    ) -> None:
        """
        args:
            inpt_dim: The number of input neurons
        kwargs:
            outp_dim: The number of output neurons, if none it will take inpt or hddn
            ctxt_dim: The number of context features, use is determined by ctxt_type
            hddn_dim: The width of each hidden block (if list, overides depth)
            num_blocks: The number of hidden blocks (overwritten by hddn_dim if list)
            n_lyr_pbk: The number of transform layers per hidden block
            act_h: The name of the activation function to apply in the hidden blocks
            act_o: The name of the activation function to apply to the outputs
            do_out: If the network has a dedicated output block
            nrm: Type of normalisation (layer or batch) in each hidden block
            inpt_nrm: Type of normalisation (layer or batch) to apply to inputs
                Useful for transformers with residual connections
            drp: Dropout probability for hidden layers (0 means no dropout)
            do_res: Use res-connections between hidden blocks (only if same size)
            ctxt_in_all: Include the ctxt tensor in all blocks, not just input
        """
        super().__init__()

        ## We store the input, hddn (list), output, and ctxt dims to query them later
        self.inpt_dim = inpt_dim
        if isinstance(hddn_dim, list):
            self.hddn_dim = hddn_dim
        else:
            self.hddn_dim = num_blocks * [hddn_dim]
        self.outp_dim = outp_dim or inpt_dim if do_out else self.hddn_dim[-1]
        self.num_blocks = len(self.hddn_dim)
        self.ctxt_dim = ctxt_dim
        self.do_out = do_out
        self.inpt_nrm = inpt_nrm

        ## Necc for this module to work with the nflows package
        self.hidden_features = self.hddn_dim[-1]

        ## Input normalisation
        if self.inpt_nrm != "none":
            self.inpt_nrm_layer = get_nrm(inpt_nrm, inpt_dim)

        ## Input MLP block
        self.input_block = MLPBlock(
            inpt_dim=self.inpt_dim,
            outp_dim=self.hddn_dim[0],
            ctxt_dim=self.ctxt_dim,
            act=act_h,
            nrm=nrm,
            drp=drp,
        )

        ## All hidden blocks as a single module list
        self.hidden_blocks = []
        if self.num_blocks > 1:
            self.hidden_blocks = nn.ModuleList()
            for h_1, h_2 in zip(self.hddn_dim[:-1], self.hddn_dim[1:]):
                self.hidden_blocks.append(
                    MLPBlock(
                        inpt_dim=h_1,
                        outp_dim=h_2,
                        ctxt_dim=self.ctxt_dim if ctxt_in_all else 0,
                        n_layers=n_lyr_pbk,
                        act=act_h,
                        nrm=nrm,
                        drp=drp,
                        do_res=do_res,
                    )
                )

        ## Output block (optional and there is no normalisation, dropout or context)
        if do_out:
            self.output_block = MLPBlock(
                inpt_dim=self.hddn_dim[-1], outp_dim=self.outp_dim, act=act_o
            )

    def forward(self, inputs: T.Tensor, ctxt: T.Tensor = None) -> T.Tensor:
        """Pass through all layers of the dense network"""

        ## Normalise the inputs to the mlp
        if self.inpt_nrm != "none":
            inputs = self.inpt_nrm_layer(inputs)

        ## Pass through the input block
        inputs = self.input_block(inputs, ctxt)

        ## Pass through each hidden block
        for h_block in self.hidden_blocks:  ## Context tensor will only be used if
            inputs = h_block(inputs, ctxt)  ## block was initialised with a ctxt dim

        ## Pass through the output block
        if self.do_out:
            inputs = self.output_block(inputs)

        return inputs

    def __repr__(self):
        string = ""
        if self.inpt_nrm != "none":
            string += "\n  (nrm): " + repr(self.inpt_nrm_layer)
        string += "\n  (inp): " + repr(self.input_block) + "\n"
        for i, h_block in enumerate(self.hidden_blocks):
            string += f"  (h-{i+1}): " + repr(h_block) + "\n"
        if self.do_out:
            string += "  (out): " + repr(self.output_block)
        return string

    def one_line_string(self):
        """Return a one line string that sums up the network structure"""
        string = ""
        if self.inpt_nrm != "none":
            string += "LN>"
        string += str(self.inpt_dim) + ">"
        string += str(self.input_block.outp_dim) + ">"
        string += ">".join(
            [
                str(layer.out_features)
                for hidden in self.hidden_blocks
                for layer in hidden.block
                if isinstance(layer, nn.Linear)
            ]
        )
        if self.do_out:
            string += ">" + str(self.outp_dim)
        return string


class DeepSet(nn.Module):
    """A deep set network that can provide attention pooling"""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        pool_type: str = "mean",
        attn_type: str = "mean",
        feat_net_kwargs=None,
        attn_net_kwargs=None,
        post_net_kwargs=None,
    ) -> None:
        """
        args:
            inpt_dim: The number of input features
            outp_dim: The number of desired output featues
        kwargs:
            ctxt_dim: Dimension of the context information for all networks
            pool_type: The type of set pooling applied; mean, sum, max or attn
            attn_type: The type of attention; mean, sum, raw
            feat_net_kwargs: Keyword arguments for the feature network
            attn_net_kwargs: Keyword arguments for the attention network
            post_net_kwargs: Keyword arguments for the post network
        """
        super().__init__()

        ## Dict default arguments
        feat_net_kwargs = feat_net_kwargs or {}
        attn_net_kwargs = attn_net_kwargs or {}
        post_net_kwargs = post_net_kwargs or {}

        ## For the attention network the default output must be set to 1
        ## The dense network default output is the same as the input
        if "outp_dim" not in attn_net_kwargs:
            attn_net_kwargs["outp_dim"] = 1

        ## Save the class attributes
        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.pool_type = pool_type
        self.attn_type = attn_type

        ## Create the feature extraction network
        self.feat_net = DenseNetwork(
            self.inpt_dim, ctxt_dim=self.ctxt_dim, **feat_net_kwargs
        )
        pooled_dim = self.feat_net.outp_dim

        ## For an attention deepset
        if self.pool_type == "attn":

            ## Create the attention network
            self.attn_net = DenseNetwork(
                self.inpt_dim, ctxt_dim=self.ctxt_dim, **attn_net_kwargs
            )

            ## Pooled dimension increases with multiheaded attention
            pooled_dim *= self.attn_net.outp_dim

        ## Create the post network to update the pooled features of the set
        self.post_net = DenseNetwork(
            pooled_dim, outp_dim, ctxt_dim=self.ctxt_dim, **post_net_kwargs
        )

    def forward(
        self, tensor: T.tensor, mask: T.BoolTensor, ctxt: Union[T.Tensor, list] = None
    ):
        """The expected shapes of the inputs are
        - tensor: batch x setsize x features
        - mask: batch x setsize
        - ctxt: batch x features
        """

        ## Combine the context information if it is a list
        if isinstance(ctxt, list):
            ctxt = smart_cat(ctxt)

        ## Pass the non_zero values through the feature network
        feat_outs = pass_with_mask(tensor, self.feat_net, mask, context=ctxt)

        ## For attention
        if self.pool_type == "attn":
            attn_outs = pass_with_mask(
                tensor,
                self.attn_net,
                mask,
                context=ctxt,
                padval=0 if self.attn_type == "raw" else -T.inf,
            )

            ## Apply either a softmax for weighted mean or softplus for weighted sum
            if self.attn_type == "mean":
                attn_outs = F.softmax(attn_outs, dim=-2)
            elif self.attn_type == "sum":
                attn_outs = F.softplus(attn_outs)

            ## Broadcast the attention to get the multiple poolings and sum
            feat_outs = (
                (feat_outs.unsqueeze(-1) * attn_outs.unsqueeze(-2))
                .flatten(start_dim=-2)
                .sum(dim=-2)
            )

        ## For the other types of pooling use the masked pool method
        else:
            feat_outs = masked_pool(self.pool_type, feat_outs, mask)

        ## Pass the pooled information through post network and return
        return self.post_net(feat_outs, ctxt)
