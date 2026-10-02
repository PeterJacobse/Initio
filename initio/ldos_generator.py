import os
import numpy as np
from . import vaspwfc, Structure
from .bands import get_eigenenergies_from_wavecar
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt



class LDOSGenerator:
    """
    Generates an object that holds information on the single-particle wavefunctions.
    This information is used to calculate simulated local density of states or scanning tunneling microscopy maps, given information of the tip.
    """
    def __init__(self, wavecar_object: vaspwfc, structure: Structure, energy_range_eV: list | np.ndarray = [], gamma_meV: float = 50, n_gammas: int = 5, tip_width_pm: float = 0., tip_p_fraction: float = 0., tip_height_pm: float = 200.):
        self.wfc = wavecar_object
        self.struc = structure
        self.set_tip_shape(tip_width_pm, tip_p_fraction)
        self.set_tip_height(tip_height_pm)
        
        # Initialize important parameters
        self.n_spins = int(wavecar_object._nspin)
        self.n_kpts = int(wavecar_object._nkpts)
        
        self.gamma_eV = gamma_meV / 1000
        self.gamma2 = self.gamma_eV ** 2
        energy_padding_eV = n_gammas * self.gamma_eV # All eigenstates within the energy padding from the energy_range will be considered
        self.voxels = wavecar_object._ngrid * 2
        voxel_size_Ang = np.diag(wavecar_object._Acell) / self.voxels # This may break if the unit cell is not cubic and organized as [x, y, z]
        self.voxels_per_pm = 1 / (100 * np.mean(voxel_size_Ang))
        self.z_nm_per_vox = voxel_size_Ang[2] / 10
        atom_z_values_nm = structure.cart_coords[:, 2] * .1
        self.z_surface_nm = np.mean(np.partition(atom_z_values_nm, -12)[-12:-10]) # Derive where the surface is from taking the 10 highest z-coordinates in the structure, omitting 2 possible outliers



        # Get the band energies and take a selection ranging from n_gammas times the Lorentzian width below the minimum energy value to n_gammas times above the maximum energy value
        energy_dict = get_eigenenergies_from_wavecar(wavecar_object)
        spin_up_energies = energy_dict["energies"]["spin up"]
        spin_down_energies = energy_dict["energies"]["spin down"]
        k_resolved_spin_up_energies = spin_up_energies.reshape(self.n_kpts, -1)
        k_resolved_spin_down_energies = spin_down_energies.reshape(self.n_kpts, -1)        
        
        min_up_index = min([int(np.where(k_resolved_spin_up_energies[kpt] > min(energy_range_eV) - energy_padding_eV)[0][0]) for kpt in range(len(k_resolved_spin_up_energies))])
        min_down_index = min([int(np.where(k_resolved_spin_down_energies[kpt] > min(energy_range_eV) - energy_padding_eV)[0][0]) for kpt in range(len(k_resolved_spin_down_energies))])
        min_orbital_index = min((min_up_index, min_down_index))
        max_up_index = max([int(np.where(k_resolved_spin_up_energies[kpt] < max(energy_range_eV) + energy_padding_eV)[0][-1]) for kpt in range(len(k_resolved_spin_up_energies))])
        max_down_index = max([int(np.where(k_resolved_spin_down_energies[kpt] < max(energy_range_eV) + energy_padding_eV)[0][-1]) for kpt in range(len(k_resolved_spin_down_energies))])
        max_orbital_index = max((max_up_index, max_down_index))
        orbital_indices = np.arange(min_orbital_index, max_orbital_index + 1, 1, dtype = np.int32)

        selected_spin_up_energies = np.concatenate([k_resolved_spin_up_energies[kpt][orbital_indices] for kpt in range(self.n_kpts)])
        selected_spin_down_energies = np.concatenate([k_resolved_spin_down_energies[kpt][orbital_indices] for kpt in range(self.n_kpts)])
        self.energies = np.concatenate((selected_spin_up_energies, selected_spin_down_energies))



        # Extract a subset of the wavefunctions from the wavecar file and store it in wfns
        print("Extracting wave functions from wavecar object...")
        self.wfns = np.zeros((self.n_spins, self.n_kpts, len(orbital_indices), self.voxels[0], self.voxels[1], self.voxels[2]), dtype = np.complex64)
        for spin_index in range(self.n_spins):
            for k_index in range(self.n_kpts):
                for index, orb_index in enumerate(orbital_indices):
                    self.wfns[spin_index, k_index, index] = wavecar_object.wfc_r(spin_index + 1, k_index + 1, orb_index + 1)
        print("Done!")



    def set_tip_shape(self, width_pm: float | None = None, p_fraction: float = 0.) -> None:
        if isinstance(width_pm, float | int): self.tip_width_pm = width_pm
        if isinstance(p_fraction, float | int): self.tip_p_fraction = float(np.clip(p_fraction, 0, 1))
        return
    
    def set_tip_height(self, height_pm: float | None = None) -> None:
        if isinstance(height_pm, float | int): self.tip_height_pm = height_pm
        return
    
    def get_maps(self, energy_values_meV: float | int | list | np.ndarray = 0., height_values_pm: float | int | list | np.ndarray | None = None,
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
                    
                    weights = self.gamma_eV / (self.gamma2 + en_differences ** 2)
                    weights /= np.sum(weights)

                    s_image = np.average(s_densities, axis = 0, weights = weights)
                    p_image = np.average(p_densities, axis = 0, weights = weights)
                    
                    for p_index, p_fraction in enumerate(p_fractions):
                        image = (1 - p_fraction) * s_image + p_fraction * p_image
                        map_array[z_index, width_index, p_index, energy_index] = image
                        if not isinstance(output_folder, str): continue
                        plt.imsave(os.path.join(output_folder, f"LDOS_h{int(z_slice_height_pm)}pm_w{int(round(width_pm))}pm_p{int(round(p_fraction * 100))}pct@{int(round(target_energy_meV))}meV.png"), image, cmap = "gray")
        return map_array