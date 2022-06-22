"""
Functions and classes used to define the learnable and invertible transformations used
"""

from copy import deepcopy
from typing import Literal
import torch as T

from nflows.transforms import (
    CompositeTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    MaskedAffineAutoregressiveTransform,
    AffineCouplingTransform,
    PiecewiseRationalQuadraticCouplingTransform,
    LULinear,
    BatchNorm,
    ActNorm,
)

from .modules import DenseNetwork
from .utils import change_kwargs_for_made


def stacked_norm_flow(
    xz_dim: int,
    ctxt_dim: int = 0,
    nstacks: int = 3,
    param_func: Literal["made", "cplng"] = "cplng",
    invrt_func: Literal["rqs", "aff"] = "aff",
    do_lu: bool = True,
    nrm: str = "none",
    net_kwargs=None,
    rqs_kwargs=None,
) -> CompositeTransform:
    """
    Create a stacked flow using a either autoregressive or coupling layers to learn the
    paramters which are then applied to elementwise invertible transforms, which can
    either be a rational quadratic spline or an affine layer.

    After each of these transforms, there can be an extra invertible
    linear layer, followed by some normalisation.

    args:
        xz_dim: The number of input X (and output Z) features
    kwargs:
        ctxt_dim: The dimension of the context feature vector
        nstacks: The number of NSF+Perm layers to use in the overall transform
        param_func: To use either autoregressive or coupling layers
        invrt_func: To use either spline or affine transformations
        do_lu: Use an invertible linear layer inbetween splines instead of permutation
        nrm: Do a scale shift normalisation inbetween splines (batch or act)
        net_kwargs: Kwargs for the network constructor (includes ctxt dim)
        rqs_kwargs: Keyword args for the invertible spline layers
    """

    ## Dictionary default arguments (also protecting dict from chaning on save)
    net_kwargs = deepcopy(net_kwargs) or {}
    rqs_kwargs = deepcopy(rqs_kwargs) or {}

    ## We add the context dimension to the list of network keyword arguments
    net_kwargs["ctxt_dim"] = ctxt_dim

    ## For MADE netwoks change kwargs from my dense format to nflows format
    if param_func == "made":
        made_kwargs, hddn_dim = change_kwargs_for_made(net_kwargs)

    ## For coupling layers we need to define a custom network maker function
    elif param_func == "cplng":

        def net_mkr(inpt, outp):
            return DenseNetwork(inpt, outp, **net_kwargs)

    ## Start the list of transforms out as an empty list
    trans_list = []

    ## Start with a mixing layer
    if do_lu:
        trans_list.append(LULinear(xz_dim))

    ## Cycle through each stack
    for i in range(nstacks):

        ## For autoregressive funcions
        if param_func == "made":
            if invrt_func == "aff":
                trans_list.append(
                    MaskedAffineAutoregressiveTransform(xz_dim, hddn_dim, **made_kwargs)
                )

            elif invrt_func == "rqs":
                trans_list.append(
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        xz_dim,
                        hddn_dim,
                        **made_kwargs,
                        **rqs_kwargs,
                    )
                )

        ## For coupling layers
        elif param_func == "cplng":

            ## Mask the transverse coordinates first
            mask = (T.arange(xz_dim) // 2 + i + 1) % 2
            # mask = (T.arange(xz_dim) + i) % 2

            if invrt_func == "aff":
                trans_list.append(AffineCouplingTransform(mask, net_mkr))

            elif param_func == "cplng" and invrt_func == "rqs":
                trans_list.append(
                    PiecewiseRationalQuadraticCouplingTransform(
                        mask,
                        net_mkr,
                        **rqs_kwargs,
                    )
                )

        ## Add the mixing layers
        if do_lu:
            trans_list.append(LULinear(xz_dim))

        ## Normalising layers (never on last layer in stack)
        if i < nstacks - 1:
            if nrm == "batch":
                trans_list.append(BatchNorm(xz_dim))
            elif nrm == "act":
                trans_list.append(ActNorm(xz_dim))

    ## Return the list of transforms combined
    return CompositeTransform(trans_list)
