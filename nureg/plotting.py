"""
A collection of of functions useful for some of the package's plotting code
"""

from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt


def plot_multi_loss(
    path: Path,
    loss_hist: dict,
    xvals: list = None,
    xlabel: str = "epoch",
    logx: bool = False,
) -> None:
    """Plot the contents of a loss history with epoch on the x-axis
    args:
        path: Where to save the output images
        loss_hist: A dictionary containing lists of loss histories
            dict[loss_name][dataset][epoch_number]
    kwargs
        xvals: A list of values for the x-axis, if None it uses arrange
        xlabel: The label for the shared x-axis in the plots
        logx: Using a log scale on the x-axis
    """

    ## Get the x-axis values using the length of the total loss in the trainset
    ## This should always be present
    if xvals is None:
        xvals = np.arange(1, len(loss_hist["total"]["train"]) + 1)

    ## Create the main figure and subplots
    fig, axes = plt.subplots(
        len(loss_hist), 1, sharex=True, figsize=(4, 4 * len(loss_hist))
    )

    ## Account for the fact that there may be a single loss
    if len(loss_hist) == 1:
        axes = [axes]

    ## Cycle though the different loss types, each on their own axis
    for axis, lnm in zip(axes, loss_hist.keys()):
        axis.set_ylabel(lnm)
        axis.set_xlabel(xlabel)

        if logx:
            axis.set_xscale("log")

        ## Plot each dataset's history ontop of each other
        for dset, vals in loss_hist[lnm].items():

            ## Skip empty loss dictionaries (sometimes we dont have valid loss)
            if not vals:
                continue

            axis.plot(xvals, vals, label=dset)

    ## Put the legend only on the top plot and save
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"))
    plt.close(fig)


def plot_multi_hists(
    path: Union[Path, str],
    data_list: Union[list, np.ndarray],
    type_labels: Union[list, str],
    col_labels: Union[list, str],
    normed: bool = False,
    bins: Union[list, str] = "auto",
    logy: bool = False,
    ylim: list = None,
    rat_ylim=(0, 2),
    rat_label=None,
    scale: int = 5,
    leg: bool = True,
    incl_zeros: bool = True,
    already_hists: bool = False,
    hist_fills: list = None,
    hist_colours: list = None,
    hist_kwargs: dict = None,
    hist_scale: float = 1,
    incl_overflow: bool = False,
    incl_underflow: bool = True,
    do_step: bool = True,
    do_ratio_to_first: bool = False,
    as_pdf: bool = False,
):
    """Plot multiple histograms given a list of 2D tensors/arrays
    - Performs the histogramming here
    - Each column the arrays will be a seperate axis
    - Matching columns in each array will be superimposed on the same axis

    args:
        path: The save location of the plots
        data_list: A list of tensors or numpy arrays
        type_labels: A list of labels for each tensor in data_list
        col_labels: A list of labels for each column/histogram
        normed: If the histograms are to be a density plot
        bins: The bins to use for each axis, can use numpy's strings
        logy: If we should use the log in the y-axis
        ylim: The y limits for all plots
        rat_ylim: The y limits of the ratio plots
        rat_label: The label for the ratio plot
        scale: The size in inches for each subplot
        leg: If the legend should be plotted
        incl_zeros: If zero values should be included in the histograms or ignored
        already_hists: If the data is already histogrammed and doesnt need to be binned
        hist_fills: Bool for each histogram in data_list, if it should be filled
        hist_colours: Color for each histogram in data_list
        hist_kwargs: Additional keyword arguments for the line for each histogram
        hist_scale: Amount to scale all histograms
        incl_overflow: Have the final bin include the overflow
        incl_underflow: Have the first bin include the underflow
        do_step: If the data should be represented as a step plot
        do_ratio_to_first: Include a ratio plot to the first histogram in the list
        as_pdf: Also save an additional image in pdf format
    """

    ## Make sure we are using a pathlib type variable
    path = Path(path)

    ## Make the arguments lists for generality
    if not isinstance(data_list, list):
        data_list = [data_list]
    if not isinstance(type_labels, list):
        type_labels = [type_labels]
    if not isinstance(col_labels, list):
        col_labels = [col_labels]
    if not isinstance(bins, list):
        bins = len(data_list[0][0]) * [bins]
    if not isinstance(hist_colours, list):
        hist_colours = len(data_list) * [hist_colours]

    ## Check the number of histograms to plot
    n_data = len(data_list)
    n_axis = len(data_list[0][0])

    ## Make sure the there are not too many subplots
    if n_axis > 20:
        raise RuntimeError("You are asking to create more than 20 subplots!")

    ## Create the figure and axes listss
    dims = np.array([n_axis, 1])
    size = np.array([n_axis, 1.0])
    if do_ratio_to_first:
        dims *= np.array([1, 2])
        size *= np.array([1, 1.2])
    fig, axes = plt.subplots(
        *dims[::-1],
        figsize=tuple(scale * size),
        gridspec_kw={"height_ratios": [3, 1] if do_ratio_to_first else {1}},
    )
    if n_axis == 1 and not do_ratio_to_first:
        axes = np.array([axes])
    axes = axes.reshape(dims)

    ## Replace the zeros
    if not incl_zeros:
        for d in data_list:
            d[d == 0] = np.nan

    ## Cycle through each axis
    for i in range(n_axis):
        b = bins[i]

        ## Reduce bins based on number of unique datapoints
        ## If the number of datapoints is less than 10 then we assume interger types
        if isinstance(b, str) and not already_hists:
            unq = np.unique(data_list[0][:, i])
            n_unique = len(unq)
            if 1 < n_unique < 10:
                b = (unq[1:] + unq[:-1]) / 2  ## Use midpoints
                b = np.append(b, unq.max() + unq.max() - b[-1])  ## Add final bin
                b = np.insert(b, 0, unq.min() + unq.min() - b[0])  ## Add initial bin

        ## Cycle through the different data arrays
        for j in range(n_data):

            ## Read the binned data from the array
            if already_hists:
                histo = data_list[j][:, i]

            ## Calculate histogram of the column and remember the bins
            else:

                ## Get the bins for the histogram based on the first plot
                if j == 0:
                    b = np.histogram_bin_edges(data_list[j][:, i], bins=b)

                ## Apply overflow and underflow (make a copy)
                data = np.copy(data_list[j][:, i])
                if incl_overflow:
                    data = np.minimum(data, b[-1])
                if incl_underflow:
                    data = np.maximum(data, b[0])

                ## Calculate the histogram
                histo, _ = np.histogram(data, b, density=normed)

            ## Apply the scaling factor
            histo = histo * hist_scale

            ## Save the first histogram for the ratio plots
            if j == 0:
                denom_hist = histo

            ## Get the additional keywork arguments
            if hist_kwargs is not None:
                kwargs = {key: val[j] for key, val in hist_kwargs.items()}
            else:
                kwargs = {}

            ## Plot the fill
            ydata = histo.tolist()
            ydata = [ydata[0]] + ydata
            if hist_fills is not None and hist_fills[j]:
                axes[i, 0].fill_between(
                    b,
                    ydata,
                    label=type_labels[j],
                    step="pre" if do_step else None,
                    alpha=0.4,
                    color=hist_colours[j],
                )

            ## Plot the histogram as a step graph
            elif do_step:
                axes[i, 0].step(
                    b, ydata, label=type_labels[j], color=hist_colours[j], **kwargs
                )

            else:
                axes[i, 0].plot(
                    b, ydata, label=type_labels[j], color=hist_colours[j], **kwargs
                )

            ## Plot the ratio plot
            if do_ratio_to_first:
                ydata = (histo / denom_hist).tolist()
                ydata = [ydata[0]] + ydata
                axes[i, 1].step(
                    b,
                    ydata,
                    color=hist_colours[j],
                    **kwargs,
                )

        ## Set the x_axis label
        if do_ratio_to_first:
            axes[i, 0].set_xticklabels([])
            axes[i, 1].set_xlabel(col_labels[i])
        else:
            axes[i, 0].set_xlabel(col_labels[i])

        ## Set the limits
        axes[i, 0].set_xlim(b[0], b[-1])
        if ylim is not None:
            axes[i, 0].set_ylim(*ylim)

        if do_ratio_to_first:
            axes[i, 1].set_xlim(b[0], b[-1])
            axes[i, 1].set_ylim(rat_ylim)

        ## Set the y scale to be logarithmic
        if logy:
            axes[i, 0].set_yscale("log")

        ## Set the y axis
        if normed:
            axes[i, 0].set_ylabel("Normalised Entries")
        elif hist_scale != 1:
            axes[i, 0].set_ylabel("a.u.")
        else:
            axes[i, 0].set_ylabel("Entries")
        if do_ratio_to_first:
            if rat_label is not None:
                axes[i, 1].set_ylabel(rat_label)
            else:
                axes[i, 1].set_ylabel(f"Ratio to {type_labels[0]}")

    ## Only do legend on the first axis
    if leg:
        axes[0, 0].legend()

    ## Save the image as a png
    fig.tight_layout()

    ## For ratio plots minimise the h_space
    if do_ratio_to_first:
        fig.subplots_adjust(hspace=0.08)

    fig.savefig(Path(path).with_suffix(".png"))
    if as_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
