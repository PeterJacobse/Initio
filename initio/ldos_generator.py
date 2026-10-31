import os
import numpy as np
from . import vaspwfc, Structure, plot
from .io import get_file_path_from_dialog, get_wavecar, get_structure
from .eigenstate import Eigenstate
from .bands import DOS_from_Lorentzians
from scipy.ndimage import gaussian_filter, sobel
import matplotlib.pyplot as plt
import nglview as nv
import plotly.graph_objects as go
from tqdm.notebook import tqdm
from typing import Literal



def get_spin_index(spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> int:
    if isinstance(spin, str):
        match spin.lower():                
            case "up": spin_index = 1
            case "down": spin_index = 0
            case _: spin_index = 2
    else:
        spin_index = spin
    return spin_index



class LDOSGenerator:
    """
    Generates an object that holds information on the single-particle eigenstates.
    This information is used to calculate simulated local density of states or scanning tunneling microscopy maps, given information of the tip.
    """
    def __init__(self, wavecar_object: vaspwfc, struct: Structure, gamma_meV: float = 50, n_gammas: float = 5., tip_width_pm: float = 0., tip_p_fraction: float = 0., tip_height_pm: float = 200.):
        self.wfc = wavecar_object
        self.struct = struct
        
        # Extract variables from wavecar
        self.n_spins = int(self.wfc._nspin)
        self.n_kpts = int(self.wfc._nkpts)
        self.n_bands = int(self.wfc._nbands)
        self.voxels = self.wfc._ngrid * 2
        self.latvecs = self.wfc._Acell
        
        self.all_energies = self.get_all_energies()
        self.spin_up_energies = self.get_all_energies(spin = "up")
        self.spin_down_energies = self.get_all_energies(spin = "down")
        
        # Structure
        voxel_size_Ang = np.diag(self.latvecs) / self.voxels # This may break if the unit cell is not cubic and organized as [x, y, z]
        self.voxels_per_pm = 1 / (100 * np.mean(voxel_size_Ang))
        self.z_nm_per_vox = voxel_size_Ang[2] / 10
        self.get_surface_height()
        
        # Instantiate other variables
        self.en_min = None
        self.en_max = None
        self.extracted_bands: list[int] = []
        self.eigenstates: list[Eigenstate] = []
        self.signatures: np.ndarray = np.empty((0, 8))
        
        # Energy
        self.set_energy_broadening(gamma_meV, units = "meV")
        self.set_energy_padding(n_gammas = n_gammas) # Eigenstates within an energy range of gamma times n_gammas are considered
        
        # Tip
        self.tip_width_pm = 0.
        self.tip_height_pm = 0.
        self.set_tip(tip_width_pm, tip_p_fraction, tip_height_pm)
        
        # Wavefunctions and orbitals
        self.psi_s = np.empty(shape = (self.voxels[0], self.voxels[1], 0), dtype = np.complex64)
        self.psi_p = np.empty_like(self.psi_s, dtype = np.complex64)
        self.s_orbs = np.empty_like(self.psi_s, dtype = np.float32)
        self.p_orbs = np.empty_like(self.psi_s, dtype = np.float32)
        
        self.psi_s_wide = np.empty_like(self.psi_s, dtype = np.complex64)
        self.psi_p_wide = np.empty_like(self.psi_s, dtype = np.complex64)
        self.s_orbs_wide = np.empty_like(self.psi_s, dtype = np.float32)
        self.p_orbs_wide = np.empty_like(self.psi_s, dtype = np.float32)
        
        self.orbs = np.empty_like(self.psi_s, dtype = np.float32)

    def __repr__(self) -> str:
        return self.info()
    
    def __getitem__(self, key) -> Eigenstate:
        return self.eigenstates[key]
    
    def __len__(self) -> int:
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

    def map_position_to_indices(self, position: np.ndarray | list = [0, 0, 0], fractional: bool = True):
        if not fractional: frac_pos = [position[index] / self.latvecs[index, index] for index in range(len(position))]
        else: frac_pos = position
        indices = [round(value) for value in self.voxels * frac_pos]
        return indices

    def show_structure(self, max_bond_length: float | None = None, width: int = 800, height: int = 600, atom_size: float = .3, bond_size: float = .22,
                       camera_type: Literal["orthographic", "perspective"] = "orthographic", flip_over: bool = False, background_color: str = "#000000",
                       show_tip_plane: bool = False, plane_rgb_color: list = [0.0, 0.8, 0.8]) -> nv.NGLWidget:
        view = plot.structure(self.struct, max_bond_length = max_bond_length, width = width, height = height, atom_size = atom_size, bond_size = bond_size, camera_type = camera_type, flip_over = flip_over, background_color = background_color)
        if show_tip_plane: self.add_plane_to_view(view, rgb_color = plane_rgb_color)
        return view

    def add_plane_to_view(self, view: nv.NGLWidget, z_above_surface_pm: float | int | None = None, rgb_color: list = [0.0, 0.8, 0.8]) -> None:
        if not z_above_surface_pm: z_above_surface_pm = self.tip_height_pm
        z_above_surface_A = z_above_surface_pm / 100
        z_surface_A = self.z_surface_nm * 10
        z_A = z_surface_A + z_above_surface_A
                
        vertices = [0, 0, z_A] + [self.latvecs[0, 0], 0, z_A] + [self.latvecs[0, 0], self.latvecs[1, 1], z_A] + [0, self.latvecs[1, 1], z_A]
        vertex_colors = rgb_color * 4
        triangle_indices = [0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2]

        view.shape.add_mesh(vertices, vertex_colors, triangle_indices, None, "My Z-Plane") # type: ignore
        return



    # Energy information
    def get_all_energies(self, spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> np.ndarray:
        spin_index = get_spin_index(spin)
        band_array = self.wfc._bands
        assert isinstance(band_array, np.ndarray)
        
        if spin_index in {0, 1} and self.n_spins > 1:
            spin_resolved_array = band_array[spin_index]
        else:
            spin_resolved_array = band_array
        return spin_resolved_array.reshape(-1)



    # Energy parameters
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
        output = {"Lorentzian width (meV)": self.gamma_meV, "Energy padding (meV)": self.n_gammas * self.gamma_meV, "Min energy (eV)": self.en_min, "Max energy (eV)": self.en_max, "Min band": min(self.extracted_bands), "Max band": max(self.extracted_bands)}
        return output



    # Tip
    def set_tip_width(self, width_pm: float | int | None = None, recompute: bool = True) -> None:
        if isinstance(width_pm, float | int) and not width_pm == self.tip_width_pm:
            self.tip_width_pm = width_pm
            self.compute_convolved_wavefunctions()
            if recompute:
                self.compute_orbitals()
        return

    def set_tip_p_fraction(self, p_fraction: float | int | None = None) -> None:
        if isinstance(p_fraction, float | int):
            self.tip_p_fraction = float(np.clip(p_fraction, 0, 1))
            self.compute_orbitals()
        return

    def set_tip_shape(self, width_pm: float | int | None = None, p_fraction: float | int | None = None, recompute: bool = True) -> None:
        self.set_tip_width(width_pm, recompute = False)
        self.set_tip_p_fraction(p_fraction)
        return

    def set_tip_height(self, height_pm: float | int | None = None, recompute: bool = True) -> None:
        if isinstance(height_pm, float | int) and not height_pm == self.tip_height_pm:
            self.tip_height_pm = height_pm            
            self.compute_2D_wavefunctions() # Wavefunctions are always sliced anew, but whether to compute the convolved orbitals depends on recompute
            if recompute:
                self.compute_convolved_wavefunctions()
                self.compute_orbitals()
        return

    def set_tip(self, width_pm: float | None = None, p_fraction: float | None = None, height_pm: float | None = None) -> None:
        self.set_tip_height(height_pm, recompute = False) # Prevent premature convolution and p-wave fraction calculation
        self.set_tip_shape(width_pm, p_fraction, recompute = False)
        return
    
    def get_tip(self) -> dict[str, float]:
        output = {"width (pm)": self.tip_width_pm, "height (pm)": self.tip_height_pm, "p_fraction": self.tip_p_fraction}
        return output

    def get_parameters(self) -> dict[str, float]:
        output = self.get_energy_parameters() | self.get_tip()
        return output

    def parameters(self) -> dict[str, float]:
        return self.get_parameters()



    # Eigenstate management. Eigenstate metadata, called 'signatures' (containing spin index, k-point, band index, energy and occupancy) are added in sync with new wavefunctions
    def add_eigenstates(self, en_min: float = 1000., en_max: float = 1000., add_energy_padding: bool = True) -> None:
        # Analyze which bands need to be added
        if add_energy_padding:
            en_min -= self.gamma_eV * self.n_gammas
            en_max += self.gamma_eV * self.n_gammas
        
        band_crosses_min = np.any(self.wfc._bands > en_min, axis = (0, 1))
        band_min = int(np.where(band_crosses_min)[0][0])
        band_crosses_max = np.any(self.wfc._bands < en_max, axis = (0, 1))
        band_max = int(np.where(band_crosses_max)[0][-1]) + 1
        
        if len(self.extracted_bands) > 1 and min(self.extracted_bands) < band_min and max(self.extracted_bands) > band_max: return # No bands to append: return



        if len(self.extracted_bands) < 1: # No previously extracted bands. Extract all of them.
            missing_bands = np.arange(band_min, band_max + 1)
            if len(missing_bands) < 1: return
            self.psi_s = np.empty(shape = (self.voxels[0], self.voxels[1], len(missing_bands) * self.n_spins * self.n_kpts), dtype = np.complex64)
            self.psi_p = np.empty(shape = (self.voxels[0], self.voxels[1], len(missing_bands) * self.n_spins * self.n_kpts), dtype = np.complex64)
            self.signatures = np.empty(shape = (len(missing_bands) * self.n_spins * self.n_kpts, 8), dtype = np.float32)
            self.eigenstates = []
            counter = 0
            
            tq = tqdm(missing_bands, desc = "Adding eigenstates and slicing wavefunctions")
            for band_index in tq:
                for spin_index in range(self.n_spins):
                    for kpoint_index in range(self.n_kpts):
                        new_eigenstate = Eigenstate.from_wavecar(self.wfc, spin = spin_index, kpoint = kpoint_index, band = int(band_index), extract_wavefunction = False) # type: ignore
                        self.eigenstates.append(new_eigenstate)
                        
                        psi_s, psi_p = self.get_2D_wavefunctions(new_eigenstate)
                        self.psi_s[:, :, counter] = psi_s
                        self.psi_p[:, :, counter] = psi_p
                        sig = new_eigenstate.get_signature()
                        self.signatures[counter] = sig
                        
                        counter += 1
                self.extracted_bands.append(int(band_index))
            tq.close()

        if len(self.extracted_bands) < 1: return # If this check is triggered, it means that extracted_bands is still empty. So no eigenstates were added in the above section. In this case, return



        if band_min < min(self.extracted_bands): # This check is safe because at this point self.extracted_bands must have elements
            bands_to_prepend = np.arange(band_min, min(self.extracted_bands))
            psi_s_to_prepend = np.empty(shape = (self.voxels[0], self.voxels[1], len(bands_to_prepend) * self.n_spins * self.n_kpts), dtype = np.complex64)
            psi_p_to_prepend = np.empty(shape = (self.voxels[0], self.voxels[1], len(bands_to_prepend) * self.n_spins * self.n_kpts), dtype = np.complex64)
            signatures_to_prepend = np.empty(shape = (len(bands_to_prepend) * self.n_spins * self.n_kpts, 8), dtype = np.float32)
            eigenstates_to_prepend = []
            counter = 0
            
            tq = tqdm(bands_to_prepend, desc = "Prepending eigenstates and slicing wavefunctions")
            for band_index in tq:
                for spin_index in range(self.n_spins):
                    for kpoint_index in range(self.n_kpts):
                        new_eigenstate = Eigenstate.from_wavecar(self.wfc, spin = spin_index, kpoint = kpoint_index, band = int(band_index), extract_wavefunction = False) # type: ignore
                        eigenstates_to_prepend.append(new_eigenstate)
                        
                        psi_s, psi_p = self.get_2D_wavefunctions(new_eigenstate)
                        psi_s_to_prepend[:, :, counter] = psi_s
                        psi_p_to_prepend[:, :, counter] = psi_p
                        sig = new_eigenstate.get_signature()
                        signatures_to_prepend[counter] = sig
                        
                        counter += 1
                self.extracted_bands.append(int(band_index))
            tq.close()
        
            self.psi_s = np.dstack((psi_s_to_prepend, self.psi_s)) # Add the new chunks
            self.psi_p = np.dstack((psi_p_to_prepend, self.psi_p))
            self.signatures = np.vstack((signatures_to_prepend, self.signatures))
            self.eigenstates[:0] = eigenstates_to_prepend
            


        if band_max > max(self.extracted_bands): # This check is safe because at this point self.extracted_bands must have elements
            bands_to_append = np.arange(max(self.extracted_bands) + 1, band_max + 1)
            psi_s_to_append = np.empty(shape = (self.voxels[0], self.voxels[1], len(bands_to_append) * self.n_spins * self.n_kpts), dtype = np.complex64)
            psi_p_to_append = np.empty(shape = (self.voxels[0], self.voxels[1], len(bands_to_append) * self.n_spins * self.n_kpts), dtype = np.complex64)
            signatures_to_append = np.empty(shape = (len(bands_to_append) * self.n_spins * self.n_kpts, 8), dtype = np.float32)
            eigenstates_to_append = []
            counter = 0
            
            tq = tqdm(bands_to_append, desc = "Appending eigenstates and slicing wavefunctions")
            for band_index in tq:
                for spin_index in range(self.n_spins):
                    for kpoint_index in range(self.n_kpts):
                        new_eigenstate = Eigenstate.from_wavecar(self.wfc, spin = spin_index, kpoint = kpoint_index, band = int(band_index), extract_wavefunction = False) # type: ignore
                        eigenstates_to_append.append(new_eigenstate)
                        
                        psi_s, psi_p = self.get_2D_wavefunctions(new_eigenstate)
                        psi_s_to_append[:, :, counter] = psi_s
                        psi_p_to_append[:, :, counter] = psi_p
                        sig = new_eigenstate.get_signature()
                        signatures_to_append[counter] = sig
                        
                        counter += 1
                self.extracted_bands.append(int(band_index))
            tq.close()
        
            self.psi_s = np.dstack((self.psi_s, psi_s_to_append)) # Add the new chunks
            self.psi_p = np.dstack((self.psi_p, psi_p_to_append))
            self.signatures = np.vstack((self.signatures, signatures_to_append))
            self.eigenstates.extend(eigenstates_to_append)
        
        self.compute_convolved_wavefunctions()
        self.compute_orbitals()
        return

    def get_2D_wavefunctions(self, eigenstate: Eigenstate, tip_height_pm: float | int | None = None) -> tuple[np.ndarray, np.ndarray]:
        slice_index = self.get_slice_index(tip_height_pm)
        
        psi_s = eigenstate.get_2D_wavefunction(axis = 2, index = slice_index)
        dpsi_dx = .125 * sobel(psi_s, axis = 1, mode = "wrap") * eigenstate.voxel_size_Ang[0]
        dpsi_dy = .125 * sobel(psi_s, axis = 0, mode = "wrap") * eigenstate.voxel_size_Ang[1]
        psi_p = dpsi_dx + 1j * dpsi_dy
        return psi_s, psi_p

    def compute_2D_wavefunctions(self, tip_height_pm: float | None = None) -> None:
        if len(self.eigenstates) < 1: return
        self.psi_s = np.zeros((self.voxels[0], self.voxels[1], len(self.eigenstates)), dtype = np.complex128)
        self.psi_p = np.zeros((self.voxels[0], self.voxels[1], len(self.eigenstates)), dtype = np.complex128)
        
        tq = tqdm(self.eigenstates, desc = "Recomputing and slicing wavefunctions")
        for index, eigenstate in enumerate(tq):
            psi_s, psi_p = self.get_2D_wavefunctions(eigenstate, tip_height_pm)
            self.psi_s[:, :, index] = psi_s
            self.psi_p[:, :, index] = psi_p
        
        self.s_orbs = np.abs(self.psi_s) ** 2
        self.p_orbs = np.abs(self.psi_p) ** 2
        return

    def compute_convolved_wavefunctions(self, tip_width_pm: float | int | None = None, recompute_wavefunctions: bool = False) -> None:
        if len(self.eigenstates) < 1: return
        if not isinstance(tip_width_pm, float | int): tip_width_pm = self.tip_width_pm
        tip_width_px = tip_width_pm * self.voxels_per_pm        
        if not hasattr(self, "psi_s") or recompute_wavefunctions: self.compute_2D_wavefunctions()
                
        self.psi_s_wide = gaussian_filter(self.psi_s, sigma = (tip_width_px, tip_width_px, 0), mode = "wrap")
        self.psi_p_wide = gaussian_filter(self.psi_p, sigma = (tip_width_px, tip_width_px, 0), mode = "wrap")
        
        self.s_orbs_wide = np.abs(self.psi_s_wide) ** 2
        self.p_orbs_wide = np.abs(self.psi_p_wide) ** 2
        return

    def compute_orbitals(self, tip_p_fraction: float | None = None, recompute_wavefunctions: bool = False) -> None:
        if len(self.eigenstates) < 1: return
        if not isinstance(tip_p_fraction, float | int): tip_p_fraction = self.tip_p_fraction
        if not hasattr(self, "psi_s_wide") or recompute_wavefunctions: self.compute_convolved_wavefunctions(recompute_wavefunctions = True)
        
        self.orbs = tip_p_fraction * self.p_orbs_wide + self.s_orbs_wide
        return



    # Maps
    def get_eigenenergies(self, spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> np.ndarray:
        spin_index = get_spin_index(spin)
        if spin_index in {0, 1}: output = np.array([eigenstate.energy for eigenstate in self.eigenstates if eigenstate.spin == spin_index]) # Filter by spin
        else: output = np.array([eigenstate.energy for eigenstate in self.eigenstates])
        return output

    def get_eigenstate_weights_at_energy(self, energy: float | int = 0, spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> np.ndarray:
        spin_index = get_spin_index(spin)
        weights = []
        
        for eigenstate in self.eigenstates:
            if spin_index in {0, 1} and not spin_index == eigenstate.spin: continue
            energy_difference2 = (energy - eigenstate.energy) ** 2
            weight = self.gamma_eV / (self.gamma2_eV + energy_difference2)
            weights.append(weight)
        
        weights = np.array(weights) / np.pi
        return weights

    def get_map(self, energy: float | int = 0, spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> np.ndarray:
        spin_index = get_spin_index(spin)
        self.add_eigenstates(energy, energy, add_energy_padding = True)
        
        weights = self.get_eigenstate_weights_at_energy(energy, spin = spin)
        
        #if spin_index == -1:
        #    map = np.zeros_like(self.eigenstates[0].psi2D_s, dtype = np.float64) # type: ignore
        #    for index, eigenstate in enumerate(self.eigenstates):
        #        if not hasattr(eigenstate, "apparent_orbital"): self.calculate_apparent_orbital(eigenstate)
        #        apparent_orbital = getattr(eigenstate, "apparent_orbital") # This is the single-particle orbital, as observed by the tip (taking the tunneling matrix element into account). It will be stored back as a new attribute of eigenstate
        #        map += weights[index] * apparent_orbital
        #else:
        match spin_index:
            case 0:
                spin_up_entries = np.where(self.signatures[:, 1] > .5)[0]
                apparent_orbitals = self.orbs[:, :, spin_up_entries]
            case 1:
                spin_down_entries = np.where(self.signatures[:, 1] < .5)[0]
                apparent_orbitals = self.orbs[:, :, spin_down_entries]
            case _:
                apparent_orbitals = self.orbs
        
        map = np.dot(apparent_orbitals, weights)
        return map

    def get_maps(self, energies: list | np.ndarray, spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> list[np.ndarray]:
        [en_min, en_max] = [min(energies), max(energies)]
        self.add_eigenstates(en_min, en_max)
        return [self.get_map(energy, spin = spin) for energy in energies]

    def show_map(self, energy: float | int, cmap: str = "Greys_r", spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> None:
        map_array = self.get_map(energy, spin = spin)
        plt.imshow(map_array, cmap = cmap)
        plt.show()
        return



    # Spectra
    def get_spatial_weights(self, grid_indices: list | np.ndarray | None = None, position: list | np.ndarray | None = None, fractional: bool = False, spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> np.ndarray:
        spin_index = get_spin_index(spin)
        match spin_index:
            case 0:
                spin_up_entries = np.where(self.signatures[:, 1] > .5)[0]
                apparent_orbitals = self.orbs[:, :, spin_up_entries]
            case 1:
                spin_down_entries = np.where(self.signatures[:, 1] < .5)[0]
                apparent_orbitals = self.orbs[:, :, spin_down_entries]
            case _:
                apparent_orbitals = self.orbs
        
        if isinstance(position, list | np.ndarray):
            indices = self.map_position_to_indices(position, fractional = fractional)
            x_index = int(indices[0])
            y_index = int(indices[1])
        elif isinstance(grid_indices, list | np.ndarray):
            x_index = int(grid_indices[0])
            y_index = int(grid_indices[1])
        else:
            raise Exception("Either a list of grid indices or a (cartesian or fractional) position needs to be provided to calculate the spatial LDOS weights")
        
        weights = apparent_orbitals[x_index, y_index]
        return weights

    def get_DOS(self, energies: list | np.ndarray = [], gamma_meV: float | None = None, spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> np.ndarray:
        if isinstance(gamma_meV, float | int) and not gamma_meV == self.gamma_meV: self.set_energy_broadening(gamma = gamma_meV, units = "meV")
        gamma_meV = self.gamma_meV
        eigenenergies = self.get_all_energies(spin = spin)
        
        DOS = DOS_from_Lorentzians(energies, centers = eigenenergies, gammas = gamma_meV / 1000)
        return DOS

    def get_LDOS(self, energies: list | np.ndarray = [], gamma_meV: float | None = None, grid_indices: list | np.ndarray | None = None, position: list | np.ndarray | None = None, fractional: bool = False, spin: Literal[-1, 0, 1, "both", "down", "up"] = "both") -> np.ndarray:
        if not isinstance(position, list | np.ndarray) and not isinstance(grid_indices, list | np.ndarray): return self.get_DOS(energies = energies, gamma_meV = gamma_meV, spin = spin)        
        if isinstance(gamma_meV, float | int) and not gamma_meV == self.gamma_meV: self.set_energy_broadening(gamma = gamma_meV, units = "meV")
        gamma_meV = self.gamma_meV
                
        min_energy = min(energies)
        max_energy = max(energies)
        self.add_eigenstates(min_energy, max_energy, add_energy_padding = True)
        
        weights = self.get_spatial_weights(grid_indices = grid_indices, position = position, fractional = fractional, spin = spin)
        eigenenergies = self.get_eigenenergies(spin = spin)
        
        print(f"weights: {len(weights) = }, {min(weights) = }, {max(weights) = }, {len(eigenenergies) = }")
        
        LDOS = DOS_from_Lorentzians(energies, centers = eigenenergies, gammas = gamma_meV / 1000, weights = weights)
        return LDOS
    
