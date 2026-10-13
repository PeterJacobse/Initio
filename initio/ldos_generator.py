import os
import numpy as np
from . import vaspwfc, Structure
from .io import get_file_path_from_dialog, get_wavecar, get_structure
from .eigenstate import Eigenstate
from .bands import DOS_from_Lorentzians
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from typing import Literal
from pymatgen.core.sites import PeriodicSite



class LDOSGenerator:
    """
    Generates an object that holds information on the single-particle wavefunctions.
    This information is used to calculate simulated local density of states or scanning tunneling microscopy maps, given information of the tip.
    """
    def __init__(self, wavecar_object: vaspwfc, structure: Structure, gamma_meV: float = 50, n_gammas: float = 5., tip_width_pm: float = 0., tip_p_fraction: float = 0., tip_height_pm: float = 200.):
        self.wfc = wavecar_object
        self.struct = structure
        
        # Extract variables from wavecar
        self.n_spins = int(self.wfc._nspin)
        self.n_kpts = int(self.wfc._nkpts)
        self.n_bands = int(self.wfc._nbands)
        self.voxels = self.wfc._ngrid * 2
        self.latvecs = self.wfc._Acell
        
        voxel_size_Ang = np.diag(self.latvecs) / self.voxels # This may break if the unit cell is not cubic and organized as [x, y, z]
        self.voxels_per_pm = 1 / (100 * np.mean(voxel_size_Ang))
        self.z_nm_per_vox = voxel_size_Ang[2] / 10
        self.get_surface_height()
        
        # Instantiate other variables
        self.en_min = None
        self.en_max = None
        self.band_min = None
        self.band_max = None
        self.eigenstates: list[Eigenstate] = []
        
        # Energy
        self.set_energy_broadening(gamma_meV, units = "meV")
        self.set_energy_padding(n_gammas = n_gammas) # Eigenstates within an energy range of gamma times n_gammas are considered
        
        # Tip
        self.tip_width_pm = 0.
        self.tip_height_pm = 0.
        self.set_tip(tip_width_pm, tip_p_fraction, tip_height_pm)

    def __repr__(self) -> str:
        return self.info()
    
    def __getitem__(self, key):
        return self.eigenstates[key]
    
    def __len__(self):
        return len(self.eigenstates)

    def info(self) -> str:
        self.struct.formula
        if not isinstance(self.en_min, int | float): result = f"LDOSGenerator object containing a structure at self.struct ({self.struct.formula}) and wavecar object at self.wfc, but no extracted eigenstates.\nparameters: {self.get_parameters()}"
        else: result = f"LDOSGenerator object containing a structure at self.struct ({self.struct.formula}) and {len(self.eigenstates)} eigenstates extracted from wavecar object self.wfc in the energy range [{self.en_min}, {self.en_max}]\nparameters: {self.get_parameters()}."
        return result

    @classmethod
    def from_vasp_folder(cls, path: str = "", gamma_meV: float = 50, n_gammas: float = 5., tip_width_pm: float = 0., tip_p_fraction: float = 0., tip_height_pm: float = 200.):
        if not path: path = get_file_path_from_dialog()
        if os.path.isfile(path): path = os.path.dirname(path)
        if not os.path.isdir(path): raise IOError(f"Invalid folder {path}.")
        
        files_in_folder = os.listdir(path)
        if not "WAVECAR" in files_in_folder: raise IOError(f"Folder {path} contains no WAVECAR file.")
        wfc = get_wavecar(os.path.join(path, "WAVECAR"))
        
        if not "CONTCAR" in files_in_folder:
            if not "POSCAR" in files_in_folder: raise IOError(f"Folder {path} contains no structure file (POSCAR or CONTCAR).")
            else: struct = get_structure(os.path.join(path, "POSCAR"))
        else: struct = get_structure(os.path.join(path, "CONTCAR"))
        
        return cls(wfc, struct, gamma_meV, n_gammas, tip_width_pm, tip_p_fraction, tip_height_pm)



    # Structural information
    def get_surface_height(self, n_atoms: int = 10, n_outliers: int = 2) -> float:
        """Find the surface height by averaging over the top n_atoms atoms, discarding the topmost n_outliers atoms as outliers.

        Args:
            n_atoms (int, optional): Defaults to 10.
            n_outliers (int, optional): Defaults to 2.

        Returns:
            float: Surface height in nm
        """
        atom_z_values_nm = self.struct.cart_coords[:, 2] * .1
        self.z_surface_nm = float(np.mean(np.partition(atom_z_values_nm, -(n_atoms + n_outliers))[-(n_atoms + n_outliers):-n_outliers]))
        return self.z_surface_nm

    def get_slice_index(self, tip_height_pm: float | None = None) -> int:
        if not isinstance(tip_height_pm, float): tip_height_pm = self.tip_height_pm
        
        z_target_nm = self.z_surface_nm + tip_height_pm / 1000
        z_slice_index = int(round(z_target_nm / self.z_nm_per_vox))
        return z_slice_index

    def find_rarest_sites(self) -> list[PeriodicSite]:
        comp = self.struct.composition
        rarest_element = min(comp.keys(), key = lambda el: comp[el])
        index_list = [index for index, site in enumerate(self.struct) if site.specie == rarest_element]
        rarest_sites = [self.struct[index] for index in index_list]
        return rarest_sites # type: ignore



    # Energy
    def set_energy_broadening(self, gamma: float = 40, units: Literal["meV", "eV"] = "meV") -> None:
        match units:
            case "meV":
                self.gamma_meV = gamma
                self.gamma_eV = gamma / 1000
            case "eV":
                self.gamma_eV = gamma
                self.gamma_meV = 1000 * gamma
            case _:
                raise Exception(f"LDOSGenerator.set_energy_broadening: Invalid units {units}")
        self.gamma2_eV = self.gamma_eV ** 2
        return
    
    def set_energy_padding(self, n_gammas: float = 3) -> None:
        self.n_gammas = n_gammas
        return

    def get_energy_parameters(self) -> dict[str, float]:
        output = {"Lorentzian width (meV)": self.gamma_meV, "Energy padding (meV)": self.n_gammas * self.gamma_meV, "Min energy (eV)": self.en_min, "Max energy (eV)": self.en_max, "Min band": self.band_min, "Max band": self.band_max}
        return output



    # Tip
    def set_tip_width(self, width_pm: float | int = 0.) -> None:
        if isinstance(width_pm, float | int) and not width_pm == self.tip_width_pm:
            self.tip_width_pm = width_pm
            self.tip_convolve_eigenstates()
        return

    def set_tip_p_fraction(self, p_fraction: float | int = 0.) -> None:
        if isinstance(p_fraction, float | int): self.tip_p_fraction = float(np.clip(p_fraction, 0, 1))
        return

    def set_tip_shape(self, width_pm: float | int | None = None, p_fraction: float | int | None = None) -> None:
        if isinstance(width_pm, float | int): self.set_tip_width(width_pm)
        if isinstance(p_fraction, float | int): self.set_tip_p_fraction(p_fraction)
        return

    def set_tip_height(self, height_pm: float | int | None = None) -> None:
        if isinstance(height_pm, float | int) and not height_pm == self.tip_height_pm:
            self.tip_height_pm = height_pm
            self.slice_eigenstates()
            self.tip_convolve_eigenstates()
        return

    def set_tip(self, width_pm: float | None = None, p_fraction: float | None = None, height_pm: float | None = None) -> None:
        self.set_tip_shape(width_pm, p_fraction)
        if isinstance(height_pm, float | int): self.set_tip_height(height_pm)
        return
    
    def get_tip(self) -> dict[str, float]:
        output = {"width (pm)": self.tip_width_pm, "height (pm)": self.tip_height_pm, "p_fraction": self.tip_p_fraction}
        return output

    def get_parameters(self) -> dict[str, float]:
        output = self.get_energy_parameters() | self.get_tip()
        return output

    def parameters(self) -> dict[str, float]:
        return self.get_parameters()



    # Eigenstate management
    def add_eigenstates(self, en_min: float = 1000., en_max: float = 1000., add_energy_padding: bool = True) -> None:
        all_energies = self.wfc._bands
                
        # Analyze which bands need to be added
        if add_energy_padding:
            en_min -= self.gamma_eV * self.n_gammas
            en_max += self.gamma_eV * self.n_gammas
        
        band_crosses_min = np.any(all_energies > en_min, axis = (0, 1))
        band_min = int(np.where(band_crosses_min)[0][0])
        band_crosses_max = np.any(all_energies < en_max, axis = (0, 1))
        band_max = int(np.where(band_crosses_max)[0][-1]) + 1
        
        states_by_key = {}
        for eigenstate in self.eigenstates:
            key = (eigenstate.band, eigenstate.spin, eigenstate.kpoint)
            states_by_key.setdefault(key, eigenstate)

        missing_bands = [
            band for band in range(band_min, band_max)
            if any((band, spin_index, kpoint) not in states_by_key
                   for spin_index in range(self.n_spins)
                   for kpoint in range(self.n_kpts))
        ]
        if missing_bands:
            for band in tqdm(missing_bands, desc = "Extracting wavefunctions"):
                for spin_index in range(self.n_spins):
                    for kpoint in range(self.n_kpts):
                        key = (band, spin_index, kpoint)
                        if key in states_by_key: continue
                        new_eigenstate = Eigenstate.from_wavecar(self.wfc, spin = spin_index, kpoint = kpoint, band = band)
                        self.tip_convolve_eigenstate(new_eigenstate)
                        states_by_key[key] = new_eigenstate

        self.eigenstates = sorted(states_by_key.values(), key = lambda state: (state.band, state.spin, state.kpoint))

        if not isinstance(self.band_min, int) or not isinstance(self.band_max, int):
            self.band_min = band_min
            self.band_max = band_max
            self.en_min = en_min
            self.en_max = en_max
        else:
            if band_min < self.band_min: self.en_min = en_min
            if band_max > self.band_max: self.en_max = en_max
            self.band_min = min(self.band_min, band_min)
            self.band_max = max(self.band_max, band_max)
        return

    def slice_eigenstate(self, eigenstate: Eigenstate, tip_height_pm: float | None = None) -> None:
        slice_index = self.get_slice_index(tip_height_pm)
        eigenstate.slice_wavefunction(slice_index, axis = 2)
        return

    def slice_eigenstates(self, tip_height_pm: float | None = None) -> None:
        [self.slice_eigenstate(eigenstate, tip_height_pm) for eigenstate in self.eigenstates]
        return

    def tip_convolve_eigenstate(self, eigenstate: Eigenstate, tip_width_pm: float | None = None) -> None:
        if not hasattr(eigenstate, "psi2D_s"): self.slice_eigenstate(eigenstate)
        assert hasattr(eigenstate, "psi2D_s") and hasattr(eigenstate, "psi2D_p")
        
        psi2D_s = eigenstate.psi2D_s
        psi2D_p = eigenstate.psi2D_p

        if not isinstance(tip_width_pm, float): tip_width_pm = self.tip_width_pm
        tip_width_px = tip_width_pm * self.voxels_per_pm
                
        psi2D_s_wide = gaussian_filter(psi2D_s, tip_width_px, mode = "wrap")
        psi2D_p_wide = gaussian_filter(psi2D_p, tip_width_px, mode = "wrap")
        
        # Store the tip convolved eigenstates as new attributes of the eigenstate object
        setattr(eigenstate, "psi2D_s_wide", psi2D_s_wide)
        setattr(eigenstate, "psi2D_p_wide", psi2D_p_wide)
        return

    def tip_convolve_eigenstates(self, tip_width_pm: float | None = None) -> None:
        if len(self.eigenstates) < 1: return
        for eigenstate in tqdm(self.eigenstates, desc = "Recomputing the tunneling matrix elements"):
            self.tip_convolve_eigenstate(eigenstate, tip_width_pm)
        return



    # Maps
    def get_eigenenergies(self, spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> np.ndarray:
        if isinstance(spin, str):
            match spin.lower():                
                case "up": spin_index = 0
                case "down": spin_index = 1
                case _: spin_index = 2
        else:
            spin_index = spin
                
        if spin_index in {0, 1}: output = np.array([eigenstate.energy for eigenstate in self.eigenstates if eigenstate.spin == spin_index])
        else: output = np.array([eigenstate.energy for eigenstate in self.eigenstates])
        return output

    def get_energy_weights(self, energy: float | int = 0, spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> np.ndarray:
        if isinstance(spin, str):
            match spin.lower():                
                case "up": spin_index = 0
                case "down": spin_index = 1
                case _: spin_index = 2
        else:
            spin_index = spin
        
        weights = []
        
        for eigenstate in self.eigenstates:
            if spin_index in {0, 1} and not spin_index == eigenstate.spin: continue
            energy_difference2 = (energy - eigenstate.energy) ** 2
            weight = self.gamma_eV / (self.gamma2_eV + energy_difference2)
            weights.append(weight)
        
        weights = np.array(weights) / np.pi
        return weights

    def get_map(self, energy: float | int = 0) -> np.ndarray:
        self.add_eigenstates(energy, energy, add_energy_padding = True)
        weights = self.get_energy_weights(energy)

        map = np.zeros_like(self.eigenstates[0].psi2D_s, dtype = np.float64) # type: ignore
        
        for index, eigenstate in enumerate(self.eigenstates):
            if not hasattr(eigenstate, "psi2D_s_wide"): self.tip_convolve_eigenstate(eigenstate)
            
            psi2D_s_wide = getattr(eigenstate, "psi2D_s_wide")
            psi2D_p_wide = getattr(eigenstate, "psi2D_p_wide")
            
            psi2D_s_abs_squared = np.abs(psi2D_s_wide) ** 2
            psi2D_p_abs_squared = np.abs(psi2D_p_wide) ** 2
            
            weighed_psi_squared = self.tip_p_fraction * psi2D_p_abs_squared + (1 - self.tip_p_fraction) * psi2D_s_abs_squared
            map += weights[index] * weighed_psi_squared
        return map

    def get_maps(self, energies: list | np.ndarray) -> list[np.ndarray]:
        return [self.get_map(energy) for energy in energies]

    def show_map(self, energy: float | int, cmap: str = "Greys_r") -> None:
        map_array = self.get_map(energy)
        plt.imshow(map_array, cmap = cmap)
        plt.show()
        return



    # Spectra
    def get_spatial_weights(self, x_nm: float = 0., y_nm: float = 0.) -> np.ndarray:
        return np.zeros((3, 3))

    def get_DOS_spectrum(self, energies: list | np.ndarray = [], gamma_meV: float = 50, spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> np.ndarray:
        min_energy = min(energies)
        max_energy = max(energies)
        self.add_eigenstates(min_energy, max_energy, add_energy_padding = True)
        eigenenergies = self.get_eigenenergies(spin = spin)
        
        DOS = DOS_from_Lorentzians(energies, centers = eigenenergies, gammas = gamma_meV / 1000)
        return DOS
    
    def get_LDOS_spectrum(self, energies: list | np.ndarray = [], spin: Literal[-1, 0, 1, "both", "down", "up"] = "both", x_nm: float = 0., y_nm: float = 0.) -> np.ndarray:
        min_energy = min(energies)
        max_energy = max(energies)
        self.add_eigenstates(min_energy, max_energy, add_energy_padding = True)        
                
        return np.zeros((3, 3)) 
    
