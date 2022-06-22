"""
A collection of functions for particle physics calulations
"""

from typing import Tuple
import numpy as np
import torch as T

from .torch_utils import empty_0dim_like


def change_cords(
    data: T.tensor, old_cords: list, new_cords: list
) -> Tuple[T.Tensor, list]:
    """Converts a tensor from spherical/to cartesian coordinates

    args:
        data: A multidimensional tensor containin the sperical components
        old_names: The current names of the coords
        new_names: The new coords to calculate
    returns:
        new_values, new_names

    Supports:
        pt, log_pt, energy, log_energy, phi, cos, sin, eta (only for 3D inputs),
        px, py, pz (only for 3D inputs)

    Makes assumptions on based on number of features in final dimension
    - 2D = pt, phi
    - 3D = pt, eta, phi
    - 4D = pt, eta, phi, energy
    - 4D+ = pt, eta, phi, energy, other (not changed, always kept)

    """

    ## Allow a string to be given which can be seperated into a list
    old_names = old_cords.split(",") if isinstance(old_cords, str) else old_cords
    new_names = new_cords.split(",") if isinstance(new_cords, str) else new_cords

    ## List of supported new names
    for new_nm in new_names:
        if new_nm not in [
            "pt",
            "log_pt",
            "energy",
            "log_energy",
            "phi",
            "cos",
            "sin",
            "eta",
            "px",
            "py",
            "pz",
        ]:
            raise ValueError(f"Unknown coordinate name: {new_nm}")

    ## Calculate the number of features in the final dimension
    n_dim = data.shape[-1]

    ## Create the slices
    pt = data[..., 0:1]
    eta = data[..., 1:2] if n_dim > 2 else empty_0dim_like(data)
    phi = data[..., 2:3] if n_dim > 2 else data[..., 1:2]
    eng = data[..., 3:4] if n_dim > 3 else empty_0dim_like(data)
    oth = data[..., 4:] if n_dim > 4 else empty_0dim_like(data)

    ## If energy is empty then we might try use pt and eta
    if eng.shape[-1] == 0 and eta.shape[-1] > 0:
        eng = pt * T.cosh(eta)

    ## A dictionary for calculating the supported new variables
    ## Using lambda functions like this prevents execution every time
    new_coord_fns = {
        "pt": lambda: pt,
        "log_pt": lambda: T.log(pt),
        "energy": lambda: eng,
        "log_energy": lambda: T.log(eng),
        "phi": lambda: phi,
        "cos": lambda: T.cos(phi),
        "sin": lambda: T.sin(phi),
        "eta": lambda: eta,
        "px": lambda: pt * T.cos(phi),
        "py": lambda: pt * T.sin(phi),
        "pz": lambda: pt * T.sinh(eta),
        "oth": lambda: oth,
    }

    ## Create a dictionary of the requested coordinates then trim non empty values
    new_coords = {key: new_coord_fns[key]() for key in new_names}
    new_coords = {key: val for key, val in new_coords.items() if val.shape[-1] != 0}

    ## Return the combined tensors and the collection of new names (with unchanged)
    new_vals = T.cat(list(new_coords.values()) + [oth], dim=-1)
    new_names = list(new_coords.keys())
    new_names += old_names[4:]

    return new_vals, new_names


def neut_to_pxpypz(data: T.Tensor, names: list, to_pxpyeta: bool = False) -> T.Tensor:
    """Convert a tensor containing neutrino information back to px, py, pz

    This is primarily for evaluating a model which is trained using any collection
    of inputs as we need a standard set

    args:
        data: The data tensor where each row represents a single neutrino
        names: The list of variables contained in final dimension of data
    kwargs:
        to_pxpyeta: If the returned tensor should have eta in the final dimension not z
    """

    ## List of supported new names
    for new_nm in names:
        if new_nm not in [
            "pt",
            "log_pt",
            "energy",
            "log_energy",
            "phi",
            "cos",
            "sin",
            "eta",
            "px",
            "py",
            "pz",
        ]:
            raise ValueError(f"Unknown coordinate name: {new_nm}")

    ## Create a dict of the co_ordinates we have access to
    cdt = {nm: data[..., names.index(nm)] for nm in names}

    ## First we need to calculate pt (used for eta)
    if "pt" in cdt.keys():
        pass
    elif "log_pt" in cdt.keys():
        cdt["pt"] = cdt["log_pt"].exp()
    elif "px" in cdt.keys() and "py" in cdt.keys():
        cdt["pt"] = T.sqrt(cdt["px"] ** 2 + cdt["py"] ** 2)
    else:
        raise ValueError(f"Can't calculate pt given neutrino variables: {names}")

    ## Calculate px
    if "px" in cdt.keys():
        pass
    elif "cos" in cdt.keys():
        cdt["px"] = cdt["pt"] * cdt["cos"]
    elif "phi" in cdt.keys():
        cdt["px"] = cdt["pt"] * T.cos(cdt["phi"])
    else:
        raise ValueError(f"Can't calculate px given neutrino variables: {names}")

    ## Calculate py
    if "py" in cdt.keys():
        pass
    elif "sin" in cdt.keys():
        cdt["py"] = cdt["pt"] * cdt["sin"]
    elif "phi" in cdt.keys():
        cdt["py"] = cdt["pt"] * T.sin(cdt["phi"])
    else:
        raise ValueError(f"Can't calculate py given neutrino variables: {names}")

    ## Calculate eta
    if to_pxpyeta:
        if "eta" in cdt.keys():
            pass
        elif "pz" in cdt.keys():
            cdt["eta"] = T.arctanh(
                cdt["pz"] / T.sqrt(1e-8 + cdt["pz"] ** 2 + cdt["pt"] ** 2)
            )
        else:
            raise ValueError(f"Can't calculate eta given neutrino variables: {names}")

    ## Calculate pz instead of eta
    else:
        if "pz" in cdt.keys():
            pass
        elif "eta" in cdt.keys():
            cdt["pz"] = cdt["pt"] * T.sinh(cdt["eta"])
        else:
            raise ValueError(f"Can't calculate pz given neutrino variables: {names}")

    ## Combine the final measurements
    return T.cat(
        [
            cdt["px"].unsqueeze(-1),
            cdt["py"].unsqueeze(-1),
            cdt["pz" if not to_pxpyeta else "eta"].unsqueeze(-1),
        ],
        dim=-1,
    )


def nu_sol_comps(
    lept_px: np.ndarray,
    lept_py: np.ndarray,
    lept_pz: np.ndarray,
    lept_e: np.ndarray,
    lept_ismu: np.ndarray,
    nu_px: np.ndarray,
    nu_py: np.ndarray,
) -> np.ndarray:
    """Calculate the components of the quadratic solution for neutrino pseudorapidity"""

    ## Constants NuRegData is in GeV!
    w_mass = 80379 * 1e-3
    e_mass = 0.511 * 1e-3
    mu_mass = 105.658 * 1e-3

    ## Create an array of the masses using lepton ID
    l_mass = np.where(lept_ismu != 0, mu_mass, e_mass)

    ## Calculate all components in the quadratic equation
    nu_ptsq = nu_px**2 + nu_py**2
    alpha = w_mass**2 - l_mass**2 + 2 * (lept_px * nu_px + lept_py * nu_py)
    a = lept_pz**2 - lept_e**2
    b = alpha * lept_pz
    ## c = alpha**2 / 4 - lept_e**2 * nu_ptsq ## Using delta instead (quicker)
    delta = lept_e**2 * (alpha**2 + 4 * a * nu_ptsq)

    comp_1 = -b / (2 * a)
    comp_2 = delta / (4 * a**2)

    ## Take the sign preserving sqrt of comp_2 due to scale
    comp_2 = np.sign(comp_2) * np.sqrt(np.abs(comp_2))
    return comp_1, comp_2


def combine_comps(
    comp_1: np.ndarray,
    comp_2: np.ndarray,
    return_eta: bool = False,
    nu_pt: np.ndarray = None,
    return_both: bool = False,
) -> np.ndarray:
    """Combine the quadiratic solutions and pick one depending on complexity and size
    args:
        comp_1: First component of the quadratic
        comp_2: Signed root of the second component of the quadratic
    kwargs:
        return_eta: If the output should be eta, otherwise pz
        nu_pt: The neutrino pt, needed only if return_eta is true
        return_both: Return both solutions
    """

    ## comp_2 is already rooted, so the real component is taken to be 0 if negative
    comp_2_real = np.where(comp_2 > 0, comp_2, np.zeros_like(comp_2))

    ## Get the two solutions
    sol_1 = comp_1 + comp_2_real
    sol_2 = comp_1 - comp_2_real

    ## If both solutions are requested
    if return_both:
        if return_eta:
            return (
                np.arctanh(sol_1 / np.sqrt(sol_1**2 + nu_pt**2 + 1e-8)),
                np.arctanh(sol_2 / np.sqrt(sol_2**2 + nu_pt**2 + 1e-8)),
            )
        return sol_1, sol_2

    ## Take the smallest solution based on magnitude
    sol = np.where(np.abs(sol_1) < np.abs(sol_2), sol_1, sol_2)

    ## Return correct variable
    if return_eta:
        return np.arctanh(sol / np.sqrt(sol**2 + nu_pt**2 + 1e-8))
    else:
        return sol
