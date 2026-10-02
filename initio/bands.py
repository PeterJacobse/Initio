import os
import numpy as np
from .VaspBandUnfolding import vaspwfc
from typing import Literal
from .io import get_eigenval
from pymatgen.io.vasp.outputs import Outcar



def DOS_from_energies(eigenenergies: list | np.ndarray = [], gamma = None, sigma = None, energy_range = None, points = None, dE: float = 0.1, weights: list | np.ndarray = []) -> np.ndarray:
    use_weights = False
    
    if isinstance(eigenenergies, list): eigenenergies = np.array(eigenenergies, dtype = float)
    if not isinstance(eigenenergies, np.ndarray): raise TypeError("No valid energy list given")
    
    if isinstance(weights, list): weights = np.array(weights, dtype = float)
    if isinstance(weights, np.ndarray) and len(weights) == len(eigenenergies): use_weights = True

    E_min = np.min(eigenenergies)
    E_max = np.max(eigenenergies)
    
    if isinstance(energy_range, list | np.ndarray):
        energy_range.sort()
        if len(energy_range) > 1:
            E_min = energy_range[0]
            E_max = energy_range[1]
    
    # Explicit specification of the number of points triggers the energy list to be composed using linspace; else, use the provided energy spacing
    if isinstance(points, int): E_list = np.linspace(E_min, E_max, points, dtype = float)
    else: E_list = np.arange(E_min, E_max + dE, dE)
    DOS = np.stack([E_list, np.zeros_like(E_list)], dtype = float)
    


    # Use Lorentzian broadening
    if isinstance(gamma, float) and gamma > 0:
        gamma2 = gamma ** 2
        
        for index, energy in enumerate(E_list):
            en_diff_list = eigenenergies - energy
            en_diff_list2 = en_diff_list ** 2
            
            if use_weights:
                for eigenenergy_index, delta_E2 in enumerate(en_diff_list2):
                    DOS[1, index] += weights[eigenenergy_index] * gamma / (gamma2 + delta_E2)
            else:
                for delta_E2 in en_diff_list2:
                    DOS[1, index] += gamma / (gamma2 + delta_E2)
    return DOS

def get_HOMO_LUMO(wavecar_object: vaspwfc) -> dict[str, float]:
    eigenstate_dict = get_eigenenergies_from_wavecar(wavecar_object)
    
    bands_up = eigenstate_dict["energies"]["spin up"]
    bands_down = eigenstate_dict["energies"]["spin down"]
    occs_up = eigenstate_dict["occupations"]["spin up"]
    occs_down = eigenstate_dict["occupations"]["spin down"]
    
    HOMO_up_index = int(np.where(occs_up < .5)[0][0])
    LUMO_up_index = HOMO_up_index + 1
    HOMO_down_index = int(np.where(occs_down < .5)[0][0])
    LUMO_down_index = HOMO_down_index + 1        
    
    HOMO_up_energy = float(bands_up[HOMO_up_index] - 1)
    HOMO_down_energy = float(bands_down[HOMO_down_index] - 1)
    LUMO_up_energy = float(bands_up[LUMO_up_index] - 1)
    LUMO_down_energy = float(bands_down[LUMO_down_index] - 1)
    
    return {"HOMO_up_index": HOMO_up_index, "HOMO_down_index": HOMO_down_index, "LUMO_up_index": LUMO_up_index, "LUMO_down_index": LUMO_down_index,
            "HOMO_up_energy": HOMO_up_energy, "HOMO_down_energy": HOMO_down_energy, "LUMO_up_energy": LUMO_up_energy, "LUMO_down_energy": LUMO_down_energy}

def get_eigenenergies_from_wavecar(wavecar_object: vaspwfc) -> dict[str, dict[str, np.ndarray]]:
    n_spins = int(wavecar_object._nspin)
    n_kpts = int(wavecar_object._nkpts)
    all_bands = wavecar_object._bands
    all_band_occs = wavecar_object._occs
    
    bands_up = []
    bands_down = []
    occs_up = []
    occs_down = []
    for kpt in range(n_kpts):
        bands_up_k: list = all_bands[0][kpt] # Retrieve bands and occupations at spin index 0 and k-point index kpt
        occs_up_k: list = all_band_occs[0][kpt]
        
        match n_spins:
            case 2: # If spin-polarized, retrieve the spin down energies from spin index 1
                bands_down_k = all_bands[1][kpt]
                occs_down_k = all_band_occs[1][kpt]
            case _: # If not spin-polarized, copy the spin up energies to the spin down energies
                bands_down_k = bands_up_k[:]
                occs_down_k = occs_up_k[:]
        
        bands_up.extend(bands_up_k)
        bands_down.extend(bands_down_k)
        occs_up.extend(occs_up_k)
        occs_down.extend(occs_down_k)

    bands_up = np.array(bands_up, dtype = np.float32)
    bands_down = np.array(bands_down, dtype = np.float32)
    occs_up = np.array(occs_up, dtype = np.float32)
    occs_down = np.array(occs_down, dtype = np.float32)

    return {"energies": {"spin up": bands_up, "spin down": bands_down}, "occupations": {"spin up": occs_up, "spin down": occs_down}}

def spin_and_occupation_resolved_DOS(wavecar_object: vaspwfc, *args, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = kwargs.pop("weights", None)
    
    energy_dict = get_eigenenergies_from_wavecar(wavecar_object)
    [bands_up, bands_down] = [energy_dict["energies"][spin] for spin in ["spin up", "spin down"]]
    [occs_up, occs_down] = [energy_dict["occupations"][spin] for spin in ["spin up", "spin down"]]
    
    LDOS_up_occ = DOS_from_energies(bands_up, weights = occs_up, *args, **kwargs)
    LDOS_up_unocc = DOS_from_energies(bands_up, weights = 1 - occs_up, *args, **kwargs)
    LDOS_down_occ = DOS_from_energies(bands_down, weights = occs_down, *args, **kwargs)
    LDOS_down_unocc = DOS_from_energies(bands_down, weights = 1 - occs_down, *args, **kwargs)
    
    return LDOS_up_occ, LDOS_up_unocc, LDOS_down_occ, LDOS_down_unocc

def get_band_indices(calculation_path: str, energy_range: list | np.ndarray) -> tuple[float, int, int]:
    try:
        if not isinstance(calculation_path, str) or not os.path.isdir(calculation_path):
            print("No valid calculation path given")
            raise Exception()
        
        eigenval = get_eigenval(os.path.join(calculation_path, "EIGENVAL"))
        outcar = Outcar(os.path.join(calculation_path, "OUTCAR"))
        
        Fermi_level = outcar.efermi if isinstance(outcar.efermi, float) else 0.
        first_spin_channel = list(eigenval.eigenvalues.keys())[0]
        gamma_bands = eigenval.eigenvalues[first_spin_channel][0]

        # Enumerate to track VASP band indices (1-indexed for standard VASP reference)
        matching_indices = []
        for idx, band in enumerate(gamma_bands, start=1):
            energy = band[0]
            if np.min(energy_range) <= energy <= np.max(energy_range):
                matching_indices.append(idx)

        if not matching_indices: return Fermi_level, 0, 0

        min_band = min(matching_indices)
        max_band = max(matching_indices)        
        return Fermi_level, min_band, max_band
    
    except Exception as e:
        raise Exception("Error trying to retrieve band data from the EIGENVAL and OUTCAR")

def clean_kpath(kpath: list | np.ndarray, crystal_type: Literal["hexagonal", "orthorhombic"]) -> np.ndarray:
    if not isinstance(kpath, list | np.ndarray):
        raise Exception(f"Invalid k-path provided to clean-kpath: {kpath}")
    
    cleaned_kpath = []
    for kpoint in kpath:
        match kpoint:
            case str() if kpoint.lower() == "gamma": cleaned_kpath.append([0, 0, 0])
            case str() if kpoint.lower() == "m": cleaned_kpath.append([.5, 0, 0])
            case str() if kpoint.lower() == "k":
                if crystal_type == "hexagonal": cleaned_kpath.append([1/3, 1/3, 0])
            case str() if kpoint.lower() == "x":
                if crystal_type == "hexagonal": cleaned_kpath.append([0, .5, .5])
                elif crystal_type == "orthorhombic": cleaned_kpath.append([.5, 0, 0])
            case str() if kpoint.lower() == "a": cleaned_kpath.append([0, 0, .5])
            case str() if kpoint.lower() == "l":
                if crystal_type == "hexagonal": cleaned_kpath.append([.5, 0, .5])            
            case str() if kpoint.lower() == "h":
                if crystal_type == "hexagonal": cleaned_kpath.append([1/3, 1/3, .5])
            case str() if kpoint.lower() == "y":
                if crystal_type == "orthorhombic": cleaned_kpath.append([0, .5, 0])
            case str() if kpoint.lower() == "z":
                if crystal_type == "orthorhombic": cleaned_kpath.append([0, 0, .5])
            case list():
                if len(kpoint) == 3 and isinstance(kpoint[0], float | int): cleaned_kpath.append(kpoint)
            case np.ndarray():
                if len(kpoint) == 3: cleaned_kpath.append(list(kpoint))
            case _:
                print(f"Ignoring unrecognized kpoint in kpath: {kpoint}")
    
    return np.array(cleaned_kpath, dtype = float)

