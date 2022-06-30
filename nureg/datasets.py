"""
Pytorch Dataset definitions of the neutrino regression samples
"""

from pathlib import Path
from joblib import dump
import h5py

import torch as T
from torch.utils.data import Dataset

from .physics import change_cords, nu_sol_comps
from .plotting import plot_multi_hists
from .utils import get_scaler
from .torch_utils import to_np


class NuRegData(Dataset):
    """Dataset class for neutrino regression with ttbar events"""

    def __init__(
        self,
        dset: str = "train",
        path: str = "SetMePlease",
        lep_vars: str = "log_pt,cos,sin,eta,log_energy",
        jet_vars: str = "log_pt,cos,sin,eta,log_energy",
        out_vars: str = "log_pt,phi,eta",
        scaler_nm: str = "standard",
        incl_quad: bool = True,
    ):
        """
        kwargs:
            dset: Which dataset to pull from, either train, test or valid
            path: The location of the datafiles
            incl_quad: If the terms of the nu_z solution should be included in misc
            lep_vars: The variables to use for the lepton kinematics (also for MET)
            jet_vars: The variables to use for the jet kinematics
            out_vars: The variables to use for the momentum outputs
            scaler_nm: The scaler name for pre-post processing the data
            variables: A dictionary showing which datasets and variables to load
        """
        super().__init__()

        print(f"\nCreating dataset: {dset}")

        ## Save attributes
        self.scaler_nm = scaler_nm

        ## Change the vars to lists and save as class attributes
        self.lep_vars = lep_vars.split(",") if isinstance(lep_vars, str) else lep_vars
        self.jet_vars = jet_vars.split(",") if isinstance(jet_vars, str) else jet_vars
        self.out_vars = out_vars.split(",") if isinstance(out_vars, str) else out_vars

        ## Make sure the dset variable is correct
        if dset not in ["test", "train"]:
            raise ValueError("Unknown dset type: ", dset)

        ## Get the list of files to use for the dataset (0 sample is reserved for test)
        if dset == "train":
            file_list = list(Path(path).glob("train*"))
        else:
            file_list = [Path(path) / "test.h5"]

        ## Raise an error if there are no files found
        if not file_list:
            raise ValueError("No files found using path: " + path)

        ## Get the group variables, dict must be copied to prevent overwite
        self.variables = {
            "MET": ["MET", "phi"],
            "leptons": ["pt", "eta", "phi", "energy", "type"],
            "misc": ["njets", "nbjets"],
            "jets": ["pt", "eta", "phi", "energy", "is_tagged"],
            "truth_neutrinos": ["pt", "eta", "phi"],
        }

        ## The data starts out as empty tensors
        self.data = {key: [] for key in self.variables}

        ## Cycle through the file list and fill up group data with lists
        print(" - loading file hdf files")
        for file in file_list:
            print(file)

            with h5py.File(file, "r") as f:
                table = f["delphes"]

                ## Fill the group_data dictionary using the group_names to get variables
                for dataset in self.data.keys():

                    ## The misc data is actually a collection of datasets
                    if dataset == "misc":
                        new_lists = [table[ds] for ds in self.variables[dataset]]
                        self.data[dataset] += map(list, zip(*new_lists))

                    ## The rest is a named tuple and we pull out specific columns
                    else:
                        self.data[dataset] += table[dataset][
                            tuple(self.variables[dataset])
                        ].tolist()

        ## Cycle through the group data and turn into pytorch tensors
        for key in self.data.keys():
            self.data[key] = T.tensor(self.data[key], dtype=T.float32)

        ## Calculate the jet mask based on existing pt
        self.jet_mask = self.data["jets"][..., 0] > 0

        ## Calculate the components of the solutions for the neutrino quadratic
        ## This is done right away so we can still be flexible in coordinates later
        print(" - calculating quadratic solutions")
        self.comp_1, self.comp_2 = nu_sol_comps(
            to_np(self.data["leptons"][:, 0] * T.cos(self.data["leptons"][:, 2])),
            to_np(self.data["leptons"][:, 0] * T.sin(self.data["leptons"][:, 2])),
            to_np(self.data["leptons"][:, 0] * T.sinh(self.data["leptons"][:, 1])),
            to_np(self.data["leptons"][:, 3]),
            to_np(self.data["leptons"][:, 4]),
            to_np(self.data["MET"][:, 0] * T.cos(self.data["MET"][:, 1])),
            to_np(self.data["MET"][:, 0] * T.sin(self.data["MET"][:, 1])),
        )
        self.comp_1 = T.from_numpy(self.comp_1)
        self.comp_2 = T.from_numpy(self.comp_2)

        ## Add the quadratic components to the misc data
        if incl_quad:
            self.data["misc"] = T.hstack(
                [
                    self.data["misc"],
                    self.comp_1.unsqueeze(-1),
                    self.comp_2.unsqueeze(-1),
                ]
            )
            self.variables["misc"] += ["comp_1", "comp_2"]

        ## Convert to specified coordinates
        print(" - converting variable coordinates")
        for key in self.data:

            ## Get the new variable names based on type
            if key in ["leptons", "MET", "partons"]:
                new_vars = self.lep_vars
            elif key == "jets":
                new_vars = self.jet_vars
            elif key == "truth_neutrinos":
                new_vars = self.out_vars
            else:
                continue

            ## Change the group data tensors and the names
            self.data[key], self.variables[key] = change_cords(
                self.data[key],
                self.variables[key],
                new_vars,
            )

            ## Make sure that the jets still adhere to a neg inf mask
            if key == "jets":
                self.data["jets"][~self.jet_mask] = -T.inf

        ## If the data has been scaled and shifted yet
        self.is_processed = False

    def plot_variables(self, path):
        """Plot some histograms showing the input, context, and target
        distributions
        """

        ## Ensure the path exists
        path.mkdir(parents=True, exist_ok=True)

        ## Add a flag if the data has been procesed
        nrm_flag = "_prcd" if self.is_processed else ""

        ## For all data in the dictionary
        for key, data in self.data.items():
            plot_multi_hists(
                Path(path, key + nrm_flag),  ## Jet data is masked
                [data] if key != "jets" else [data[self.jet_mask]],
                [key],
                self.variables[key],
            )

    def save_preprocess_info(self, path: str) -> dict:
        """Return and save a dict of sklearn scalers which can preprocess the data"""
        print(" - calculating and saving preprocessing scalers")

        scalers = {}
        for key, data in self.data.items():

            scalers[key] = get_scaler(self.scaler_nm)

            ## Jet data must be masked
            if key == "jets":
                scalers[key].fit(data[self.jet_mask])  ## Jet data is masked
            else:
                scalers[key].fit(data)

        ## Save the scalers
        dump(scalers, path)

        return scalers

    def apply_preprocess(self, scalers: dict) -> None:
        """Apply the pre-processing steps to the stored data
         - Will break if the required named scalers are not present in the dictionary

        args:
            stats: dict containing named tensors describing how to prepare the data
        """
        print(" - applying preprocessing scalers")
        for key, data in self.data.items():
            ## Jet data must be masked
            if key == "jets":
                self.data[key][self.jet_mask] = T.from_numpy(
                    scalers[key].transform(data[self.jet_mask])  ## mask the jets
                ).float()
            else:
                self.data[key] = T.from_numpy(scalers[key].transform(data)).float()

        ## Make sure that jet data still adheres to the mask after transform
        self.data["jets"][~self.jet_mask] = -T.inf

        ## Change the processed state and change the names of the columns
        self.is_processed = True
        for key, names in self.variables.items():
            self.variables[key] = [nm + "_prcd" for nm in names]

    def __len__(self):
        return len(self.data["MET"])

    def __getitem__(self, idx):
        return [data[idx] for _, data in self.data.items()]

    def get_dim(self):
        """Return the dimensions of the sample returned by the getitem functon"""
        return [s.shape[-1] for s in self[0]]
