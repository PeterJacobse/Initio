import os
import numpy as np
from . import vaspwfc, Structure
from .bands import get_eigenenergies_from_wavecar, Eigenstate
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt
from typing import Literal
from tqdm.notebook import tqdm



class LDOSGenerator:
    """
    Generates an object that holds information on the single-particle wavefunctions.
    This information is used to calculate simulated local density of states or scanning tunneling microscopy maps, given information of the tip.
    """
    def __init__(self, wavecar_object: vaspwfc, structure: Structure,
                 gamma_meV: float = 50, n_gammas: float = 5.,
                 tip_width_pm: float = 0., tip_p_fraction: float = 0., tip_height_pm: float = 200.):
        self.wfc = wavecar_object
        self.struct = structure
        
        # Extract variables from wavecar
        self.n_spins = int(wavecar_object._nspin)
        self.n_kpts = int(wavecar_object._nkpts)
        self.n_bands = int(wavecar_object._nbands)
        self.voxels = wavecar_object._ngrid * 2
        
        voxel_size_Ang = np.diag(wavecar_object._Acell) / self.voxels # This may break if the unit cell is not cubic and organized as [x, y, z]
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
        if not isinstance(self.en_min, int | float): result = "Empty LDOSGenerator object"
        else: result = f"LDOSGenerator object containing {len(self.eigenstates)} eigenstates in the energy range [{self.en_min}, {self.en_max}]\nparameters: {self.get_parameters()}"
        return result
    
    def __getitem__(self, key):
        return self.eigenstates[key]
    
    def __len__(self):
        return len(self.eigenstates)



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
    def set_tip_width(self, width_pm: float = 0.) -> None:
        if isinstance(width_pm, float | int) and not width_pm == self.tip_width_pm:
            self.tip_width_pm = width_pm
            self.tip_convolve_eigenstates()
        return

    def set_tip_p_fraction(self, p_fraction: float = 0.) -> None:
        if isinstance(p_fraction, float | int): self.tip_p_fraction = float(np.clip(p_fraction, 0, 1))
        return

    def set_tip_shape(self, width_pm: float | None = None, p_fraction: float | None = None) -> None:
        if isinstance(width_pm, float): self.set_tip_width(width_pm)
        if isinstance(p_fraction, float): self.set_tip_p_fraction(p_fraction)
        return

    def set_tip_height(self, height_pm: float | None = None) -> None:
        if isinstance(height_pm, float | int) and not height_pm == self.tip_height_pm:
            self.tip_height_pm = height_pm
            self.tip_convolve_eigenstates()
        return

    def set_tip(self, width_pm: float | None = None, p_fraction: float | None = None, height_pm: float | None = None) -> None:
        self.set_tip_shape(width_pm, p_fraction)
        if isinstance(height_pm, float): self.set_tip_height(height_pm)
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
        
        eigenstate.psi2D_s_wide = psi2D_s_wide # type: ignore
        eigenstate.psi2D_p_wide = psi2D_p_wide # type: ignore
        return

    def tip_convolve_eigenstates(self, tip_width_pm: float | None = None) -> None:
        if len(self.eigenstates) < 1: return
        for eigenstate in tqdm(self.eigenstates, desc = "Recomputing the tunneling matrix elements"):
            self.tip_convolve_eigenstate(eigenstate, tip_width_pm)
        return



    # Maps
    def get_eigenenergies(self) -> np.ndarray:
        return np.array([eigenstate.energy for eigenstate in self.eigenstates])

    def get_energy_weights(self, energy: float | int = 0) -> np.ndarray:
        weights = []
        
        for eigenstate in self.eigenstates:
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
            
            psi2D_s_wide = eigenstate.psi2D_s_wide # type: ignore
            psi2D_p_wide = eigenstate.psi2D_p_wide # type: ignore
            
            psi2D_s_abs_squared = np.abs(psi2D_s_wide) ** 2
            psi2D_p_abs_squared = np.abs(psi2D_p_wide) ** 2
            
            weighed_psi_squared = self.tip_p_fraction * psi2D_p_abs_squared + (1 - self.tip_p_fraction) * psi2D_s_abs_squared
            map += weights[index] * weighed_psi_squared
        return map

    def get_maps(self, energies: list | np.ndarray) -> list[np.ndarray]:
        return [self.get_map(energy) for energy in energies]

    def get_maps_old(self, energy_values_meV: float | int | list | np.ndarray = 0., height_values_pm: float | int | list | np.ndarray | None = None,
                    width_values_pm: float | int | list | np.ndarray | None = None, p_fractions: float | int | list | np.ndarray | None = None, output_folder: str | None = None) -> np.ndarray:
        # Create the output directory relative to the calculation folder
        if isinstance(output_folder, str): os.makedirs(output_folder, exist_ok = True)
        
        # Cleaning energy and tip shape inputs
        if isinstance(height_values_pm, int | float): height_values_pm = [height_values_pm] # If a single height value is passed, put it in a list
        if not isinstance(height_values_pm, list | np.ndarray): height_values_pm = [self.tip_height_pm] # If no height values are passed, use the one saved as attribute of LDG
        
        if isinstance(p_fractions, int | float): p_fractions = [p_fractions] # If a single p fraction value is passed, put it in a list
        if not isinstance(p_fractions, list | np.ndarray): p_fractions = [self.tip_p_fraction] # If no p fraction values are passed, use the one saved as attribute of LDG
        
        if isinstance(width_values_pm, int | float): width_values_pm = [width_values_pm] # If a single height value is passed, put it in a list
        if not isinstance(width_values_pm, list | np.ndarray): width_values_pm = [self.tip_width_pm] # If no width values are passed, use the one saved as attribute of LDG
        
        if isinstance(energy_values_meV, int | float): energy_values_meV = [energy_values_meV]
        if not isinstance(energy_values_meV, list | np.ndarray): # Energies are the only parameters that have to be passed explicitly; there is no self.energy to fall back to
            print("Invalid energy value(s)")
            return
        
        
        
        map_array = np.empty(shape = (len(height_values_pm), len(width_values_pm), len(p_fractions), len(energy_values_meV), self.voxels[0], self.voxels[1]), dtype = np.float32)
        # Loop over heights
        for z_index, z_slice_height_pm in enumerate(height_values_pm):
            # Slice out the 2D wavefunction from the 3D wavefunction at the requested height
            z_target = self.z_surface_nm + z_slice_height_pm / 1000
            z_slice_index = int(round(z_target / self.z_nm_per_vox))
            wfns2D = self.wfns[:, :, :, :, :, z_slice_index]
            s_wfns = wfns2D.reshape(-1, self.voxels[0], self.voxels[1]) # Flatten out the k and spin
            p_wfns = [sobel(wavefunction, axis = 1, mode = "wrap") + 1j * sobel(wavefunction, axis = 0, mode = "wrap") for wavefunction in s_wfns]

            for width_index, width_pm in enumerate(width_values_pm):
                # Broaden the wavefunction according to their overlap with the Gaussian tip wavefunction
                width_px = width_pm * self.voxels_per_pm # Convert the width from units of picometers to voxels
                
                s_wfns_broadened = [gaussian_filter(wavefunction, width_px, mode = "wrap") for wavefunction in s_wfns]
                s_densities = np.asarray(np.abs(np.array(s_wfns_broadened)) ** 2, dtype = np.float32)
                p_wfns_broadened = [gaussian_filter(wavefunction, width_px, mode = "wrap") for wavefunction in p_wfns]
                p_densities = np.asarray(np.abs(np.array(p_wfns_broadened)) ** 2, dtype = np.float32)

                for energy_index, target_energy_meV in enumerate(energy_values_meV):
                    en_differences = np.array(self.energies, dtype = np.float32) - (.001 * target_energy_meV)
                    
                    weights = self.gamma_eV / (self.gamma2_eV + en_differences ** 2)
                    weights /= np.sum(weights)

                    s_image = np.average(s_densities, axis = 0, weights = weights)
                    p_image = np.average(p_densities, axis = 0, weights = weights)
                    
                    for p_index, p_fraction in enumerate(p_fractions):
                        image = (1 - p_fraction) * s_image + p_fraction * p_image
                        map_array[z_index, width_index, p_index, energy_index] = image
                        if not isinstance(output_folder, str): continue
                        plt.imsave(os.path.join(output_folder, f"LDOS_h{int(z_slice_height_pm)}pm_w{int(round(width_pm))}pm_p{int(round(p_fraction * 100))}pct@{int(round(target_energy_meV))}meV.png"), image, cmap = "gray")
        return map_array



    # Spectra
    def get_spatial_weights(self, x_nm: float = 0., y_nm: float = 0.) -> np.ndarray:
        return np.zeros((3, 3))

    def get_spectrum(self, x_nm: float = 0., y_nm: float = 0.) -> np.ndarray:
        return np.zeros((3, 3))
