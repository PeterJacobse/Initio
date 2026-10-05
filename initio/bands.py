import os
import numpy as np
from .VaspBandUnfolding import vaspwfc
from typing import Literal
from .io import get_eigenval
from pymatgen.io.vasp.outputs import Outcar
from scipy.ndimage import sobel



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

def DOS_from_Lorentzians(energy: float | list | np.ndarray, centers: list | np.ndarray = [], gammas: list | np.ndarray = [], weights: list | np.ndarray | None = None):
    """Expands the (Local) Density Of States from a sum of Lorentzians.

    Args:
        energy (float | list | np.ndarray): Energy where to evaluate the LDOS. Providing a np.linspace gives a DOS evaluated over that domain.
        centers (list | np.ndarray, optional): Energy values of the eigenstates. Defaults to [].
        gammas (list | np.ndarray, optional): HWHM values of the peaks. Defaults to [].
        weights (list | np.ndarray, optional): Weights of the peaks. Defaults to [].

    Returns:
        _type_: _description_
    """
    en = np.asarray(energy)
    rho = np.zeros_like(en, dtype = np.float64)
    if not isinstance(weights, np.ndarray | list): weights = np.ones_like(centers)
    
    for center, gamma, weight in zip(centers, gammas, weights):
        rho += weight * gamma / ((en - center) ** 2 + gamma ** 2)
    return rho / np.pi

def Fourier_spectrum_from_Lorentzians(V_dc: float, V_ac, V_centers, gammas, weights, N_harmonic):
    """
    Computes numerically stable physical Fourier amplitudes C_n = I(t) harmonic components
    at a fixed V_dc. Guaranteed not to blow up for high harmonics n.
    """
    V_centers = np.asarray(V_centers)
    gammas = np.asarray(gammas)
    weights = np.asarray(weights)
    
    delta_V = (V_centers - 1j * gammas) - V_dc
    rad = np.sqrt(delta_V ** 2 - V_ac ** 2)
    u = (delta_V - rad) / V_ac
    
    # CRITICAL FIX: Ensure |u| < 1 for every pole. If NumPy's branch cut chose the root with |u| > 1, take its reciprocal 1/u.
    u = np.where(np.abs(u) > 1.0, 1.0 / u, u)
    
    harmonics = np.zeros(N_harmonic + 1, dtype = np.float64)
    u_pow = np.ones_like(delta_V, dtype = np.complex128)
    
    # Compute physical Fourier amplitudes I_n
    for n in range(N_harmonic + 1):
        factor = 1.0 if n == 0 else 2.0
        
        pole_terms = u_pow / rad
        I_n = (factor / np.pi) * np.sum(weights * np.imag(pole_terms))
        harmonics[n] = I_n
        
        # Advance power: u^n -> u^(n+1) (exponentially decays since |u| < 1)
        u_pow *= u
    return harmonics

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



class Eigenstate:
    """Eigenstate object, comprising both the single-particle wavefunction and the single-particle energy (eigenenergy).
    Unlike the pyvaspwfc convention of 1-indexing (e.g. ispin = 1 or 2), Eigenstate uses 0-based indexing like Python.
    """
    def __init__(self, energy: float, spin: int | str | bool = 0, kpoint: int = 0, band: int = 0, psi: np.ndarray = np.zeros((3, 3, 3))):
        if not psi.ndim == 3: raise Exception(f"Invalid dimensionality {psi.ndim} for eigenstate wavefunction")
        self.clean_spin(spin) # Creates attributes self.spin and self.spin_name
        self.energy = energy
        self.eigenenergy = energy # Alias
        self.kpoint = kpoint
        self.band = band
        self.psi = psi
   
    def __repr__(self) -> str:
        result = f"Eigenstate object\n  Energy:\t{self.eigenenergy = } eV\n  Spin:\t\t{self.spin = }\n\t\t{self.spin_name = }\n  Kpoint:\t{self.kpoint = }\n  Band index:\t{self.band = }\n  Wavefunction:\tself.psi = <np.{self.psi.__class__.__name__} (shape = {self.psi.shape})>"
        return result
    
    def clean_spin(self, spin) -> None:
        match spin:
            case int() | bool() | float():
                if not int(spin) in {0, 1}:
                    raise Exception(f"Invalid spin index {spin}. Only 0 and 1 are recognized.")
                else:
                    self.spin = int(spin)
                    self.spin_name = "up" if int(spin) == 1 else "down"
            case str():
                if not spin.lower() in {"up", "down"}:
                    raise Exception(f"Invalid spin name {spin}. Only \"up\" and \"down\" are recognized.")
                else:
                    self.spin_name = spin.lower()
                    self.spin = 1 if spin.lower() == "up" else 0
            case _:
                raise TypeError(f"Invalid type {type(spin)} for spin provided")
        return

    def slice_wavefunction(self, index: int = 0, axis: Literal[0, 1, 2] = 0) -> None:
        match axis:
            case 0: wfn_2D = self.psi[index]
            case 1: wfn_2D = self.psi[:, index]
            case 2: wfn_2D = self.psi[:, :, index]
        
        self.psi2D_s = wfn_2D
        self.psi2D_p = sobel(wfn_2D, axis = 1, mode = "wrap") + 1j * sobel(wfn_2D, axis = 0, mode = "wrap")
        return

    @classmethod
    def from_wavecar(cls, wavecar_object: vaspwfc, spin: int = 0, kpoint: int = 0, band: int = 0, target_energy: float | None = None):
        wfc = wavecar_object
        n_kpoints = int(wfc._nkpts)
        n_spins = int(wfc._nspin)
        n_bands = int(wfc._nbands)
        
        if not isinstance(spin, int) or not spin in {0, 1}: raise Exception(f"Invalid spin index {spin}. Only 0 and 1 are recognized.")
        if spin == 1 and n_spins < 2: raise Exception(f"Invalid spin index {spin} for a wavecar object that has no spin polarization ({n_spins = }).")
        if not isinstance(kpoint, int) or kpoint > n_kpoints - 1: raise Exception(f"Invalid k-point index {kpoint} for a wavecar object containing {n_kpoints} k-points.")
        if not isinstance(target_energy, float):
            if not isinstance(band, int) or band > n_bands - 1: raise Exception(f"Invalid band index {band} for a wavecar object containing {n_bands} bands.")
        
        try:
            all_energies = wfc._bands
            assert isinstance(all_energies, np.ndarray)
            energies_at_spin_and_kpoint = all_energies[spin, kpoint]
            
            if isinstance(target_energy, float): # If a target energy is provided, find the eigenstate closest to that energy                
                energy_differences = np.abs(all_energies - target_energy)
                band = int(np.argmin(energy_differences))
            
            eigenenergy = float(energies_at_spin_and_kpoint[band])
        except Exception as e:
            raise Exception(f"Unable to extract the eigenenergy from the wavecar object: {e}")
        
        try:
            psi = wfc.wfc_r(ispin = spin + 1, ikpt = kpoint + 1, iband = band + 1)
            assert isinstance(psi, np.ndarray)
        except Exception as e:
            raise Exception(f"Unable to obtain the wavefunction from the wavecar object: {e}")
            
        return cls(eigenenergy, spin, kpoint, band, psi)

