import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors as cols
import numpy as np
from . import vaspwfc
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.io import vasp, ase
import nglview as nv
from skimage import measure
from ase.neighborlist import natural_cutoffs, NeighborList
from scipy.ndimage import gaussian_filter, zoom, sobel



class Initio:
    def __init__(self):
        pass

    def read_vasp_file(self, path: str) -> vaspwfc | Structure | bool:
        base_name = os.path.basename(path)
        
        match base_name:
            case "WAVECAR":
                wfc = self.get_wavecar(path)
                
                return wfc
            case "POSCAR" | "CONTCAR":
                struc = self.get_structure(path)
                return struc
            case _:
                print(f"Could not determine file type of provided file {path}")
                return False

    def get_wavecar(self, path: str) -> vaspwfc:
        try:
            wfc = vaspwfc(path, lgamma = False)
            
            n_kpts = int(wfc._nkpts)
            if n_kpts < 2: wfc = vaspwfc(path, lgamma = True)
            
            return wfc
        except Exception as e:
            print("Error loading the wavecar")
            return False

    def get_structure(self, path: str) -> Structure:
        try:
            structure = Structure.from_file(path)
            return structure
        except Exception as e:
            print(f"Error loading the structure: {e}")
            return False

    def DOS_from_energies(self, eigenenergies: list | np.ndarray = [], gamma = None, sigma = None, energy_range = None, points = None, dE: float = 0.1, weights: list | np.ndarray = []) -> np.ndarray:
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
        
        if isinstance(points, int): # Explicit specification of the number of points triggers the energy list to be composed using linspace
            E_list = np.linspace(E_min, E_max, points, dtype = float)
        else: # Use energy spacing instead
            E_list = np.arange(E_min, E_max + dE, dE)
            
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

    def get_HOMO_LUMO(self, wavecar_object: vaspwfc) -> dict:
        eigenstate_dict = self.get_eigenenergies_from_wavecar(wavecar_object)
        
        bands_up = eigenstate_dict["energies"]["spin up"]
        bands_down = eigenstate_dict["energies"]["spin down"]
        occs_up = eigenstate_dict["occupations"]["spin up"]
        occs_down = eigenstate_dict["occupations"]["spin down"]
        
        LUMO_up_index = int(np.where(occs_up < .5)[0][0])
        HOMO_up_index = LUMO_up_index - 1
        LUMO_down_index = int(np.where(occs_down < .5)[0][0])
        HOMO_down_index = LUMO_down_index - 1
        
        HOMO_up_energy = float(bands_up[HOMO_up_index])
        HOMO_down_energy = float(bands_down[HOMO_down_index])
        LUMO_up_energy = float(bands_up[LUMO_up_index])
        LUMO_down_energy = float(bands_down[LUMO_down_index])
        
        return {"HOMO_up_index": HOMO_up_index, "HOMO_down_index": HOMO_down_index, "LUMO_up_index": LUMO_up_index, "LUMO_down_index": LUMO_down_index,
                "HOMO_up_energy": HOMO_up_energy, "HOMO_down_energy": HOMO_down_energy, "LUMO_up_energy": LUMO_up_energy, "LUMO_down_energy": LUMO_down_energy}

    def get_eigenenergies_from_wavecar(self, wavecar_object: vaspwfc) -> dict:
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

    def spin_and_occupation_resolved_DOS(self, wavecar_object: vaspwfc, *args, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weights = kwargs.pop("weights", None)
        
        energy_dict = self.get_eigenenergies_from_wavecar(wavecar_object)
        [bands_up, bands_down] = [energy_dict["energies"][spin] for spin in ["spin up", "spin down"]]
        [occs_up, occs_down] = [energy_dict["occupations"][spin] for spin in ["spin up", "spin down"]]
        
        LDOS_up_occ = self.DOS_from_energies(bands_up, weights = occs_up, *args, **kwargs)
        LDOS_up_unocc = self.DOS_from_energies(bands_up, weights = 1 - occs_up, *args, **kwargs)
        LDOS_down_occ = self.DOS_from_energies(bands_down, weights = occs_down, *args, **kwargs)
        LDOS_down_unocc = self.DOS_from_energies(bands_down, weights = 1 - occs_down, *args, **kwargs)
        
        return (LDOS_up_occ, LDOS_up_unocc, LDOS_down_occ, LDOS_down_unocc)

    def DOS_plot(self, wavecar_object: vaspwfc, *args, **kwargs) -> plt.Figure:
        colors = kwargs.pop("colors", None)
        
        # No colors given. Use defaults
        if not isinstance(colors, list) or len(colors) < 2: colors = ["#A00000", "#0000A0"]
        # Invalid colors given. Use defaults
        if not cols.is_color_like(colors[0]): colors = ["#A00000", "#0000A0"]

        col_up_occ = list(cols.to_rgb(colors[0]))
        col_up_unocc = [.5 + .5 * channel for channel in col_up_occ]
        col_down_occ = list(cols.to_rgb(colors[1]))
        col_down_unocc = [.5 + .5 * channel for channel in col_down_occ]
        
        (LDOS_up_occ, LDOS_up_unocc, LDOS_down_occ, LDOS_down_unocc) = self.spin_and_occupation_resolved_DOS(wavecar_object, *args, **kwargs)
        
        fig, ax = plt.subplots()
        fig.set_size_inches(3, 4.6)
        ax.fill_betweenx(LDOS_up_occ[0], LDOS_up_occ[1], color = col_up_occ)
        ax.fill_betweenx(LDOS_up_unocc[0], LDOS_up_unocc[1], color = col_up_unocc)
        ax.fill_betweenx(LDOS_down_occ[0], -LDOS_down_occ[1], color = col_down_occ)
        ax.fill_betweenx(LDOS_down_unocc[0], -LDOS_down_unocc[1], color = col_down_unocc)
        
        ax.set_xlabel("DOS up (a.u.)    DOS down (a.u.)")
        ax.set_ylabel("energy (eV)")
        ax.set_xticks([])
        
        en_range = kwargs.get("energy_range")
        ax.set_ylim(en_range[0], en_range[1])
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(.1))
        
        ax.grid(True, which = "both", axis = "y", color = "gray", linewidth = 0.5, alpha = 0.5)
        return fig

    def structure_plot(self, structure: Structure, max_bond_length: float = None, width: int = 800, height: int = 600, atom_size: float = .3, bond_size: float = .22, camera_type: str = "orthographic") -> nv.NGLWidget:
        atoms = ase.AseAtomsAdaptor.get_atoms(structure)
        Z = list(structure.atomic_numbers)
        R = structure.cart_coords
        Zcolors = [.5 * rgb if atomic_number > 0 else (0, 0, 0) for atomic_number, rgb in enumerate(jmol_colors)]
        Zcolors[6] = "#404040"
        
        if not max_bond_length:
            if 74 in Z: max_bond_length = 2.6 # Shortcut for working with TMDs
            else: max_bond_length = 1.6 # Shortcut fallback for organic stuff

        cutoffs = [max_bond_length / 2.0] * len(atoms)
        nl = NeighborList(cutoffs, skin = 0.0, bothways = True, self_interaction = False)
        nl.update(atoms)

        view = nv.show_ase(atoms)
        view.stage.set_parameters(depth_of_field = 0, fog_near = 100, fog_far = 100, camera_type = camera_type)
        
        bonds = []
        for atom_index in range(len(atoms)):
            Z1 = Z[atom_index]
            R1 = R[atom_index]
            color1 = Zcolors[Z1]
            
            neighbor_indices, offsets = nl.get_neighbors(atom_index)
            for (neighbor_index, offset) in zip(neighbor_indices, offsets):
                if neighbor_index < atom_index or np.any(offset != 0): continue
                
                Z2 = Z[neighbor_index]
                R2 = R[neighbor_index]
                bond_color = color1 + Zcolors[Z2]
                
                bonds.append(["cylinder", R1, R2, bond_color, bond_size, "bond"])
        view._add_shape(bonds)
        
        view.clear_representations()
        view.component_0.add_spacefill(radiusType = "vdw", radiusScale = atom_size)
        view.component_0.add_spacefill(selection = "_C", radiusType = "vdw", radiusScale = atom_size, colorValue = Zcolors[6])
        view.component_0.add_spacefill(selection = "_N", radiusType = "vdw", radiusScale = atom_size + .1, colorValue = Zcolors[8]) # Emphasize nitrogen
        view.control.center(np.mean(structure.cart_coords, axis = 0))

        view.height = f"{height}px"
        view.width = f"{width}px"
        view.layout.height = f"{height}px"
        view.layout.width = f"{width}px"
        return view

    def orbital_plot(self, wavecar_object: vaspwfc, ispin: int = 1, ikpt: int = 1, iband: int = 1, isolevel: float = .1, opacity: float = 1., flip_x: bool = False, flip_y: bool = False, flip_z: bool = False, upsampling: int = 1,
                     structure: Structure = None, max_bond_length: float = 2.6, atom_size: float = .3, bond_size: float = .22, struc_opacity: float = 1.,
                     width: int = 800, height: int = 600, camera_type: str = "orthographic") -> nv.NGLWidget:
        if not isinstance(wavecar_object, vaspwfc):
            print(f"Invalid wave function")
            return
        if not isinstance(opacity, float | int) or opacity < 0 or opacity > 1: opacity = 1.
        if not isinstance(struc_opacity, float | int) or struc_opacity < 0 or struc_opacity > 1: struc_opacity = 1.
        
        try:
            psi: np.ndarray = zoom(wavecar_object.wfc_r(ispin = ispin, ikpt = ikpt, iband = iband), zoom = upsampling, order = 3)
            if flip_x: psi = np.flip(psi, axis = 0)
            if flip_y: psi = np.flip(psi, axis = 1)
            if flip_z: psi = np.flip(psi, axis = 2)
            orb_plus = np.abs(np.clip(psi, a_min = 0, a_max = np.inf)) ** 2
            orb_minus = np.abs(np.clip(psi, a_min = -np.inf, a_max = 0) ** 2)
        
            cell_size_Ang = wavecar_object._Acell
            voxels = wavecar_object._ngrid * 2 * upsampling
            
            voxel_size = np.diag(cell_size_Ang) / voxels
        except Exception as e:
            print(f"{e}")
            return
        
        if isinstance(structure, Structure):
            view = self.structure_plot(structure, max_bond_length, width, height, atom_size, bond_size, camera_type)
            view.update_representation(component = len(view._ngl_component_names) - 2, repr_index = 0, opacity = struc_opacity, transparent = True, depthWrite = False)
            view.update_representation(component = len(view._ngl_component_names) - 1, repr_index = 0, opacity = struc_opacity, transparent = True, depthWrite = False)
        else:
            view = nv.NGLWidget()
        
        try:
            for orb, color in zip([orb_plus, orb_minus], [[.8, .4, 0], [0, .2, .9]]):
                (verts, faces, normals, values) = measure.marching_cubes(orb, level = isolevel * np.max(orb_plus))
                verts_Ang = verts * voxel_size
                
                v0 = verts_Ang[faces[:, 0]]
                v1 = verts_Ang[faces[:, 1]]
                v2 = verts_Ang[faces[:, 2]]
                face_normals = np.cross(v1 - v0, v2 - v0)
                norms = np.linalg.norm(face_normals, axis = 1, keepdims = True)
                face_normals = np.divide(face_normals, norms, out = np.zeros_like(face_normals), where = norms != 0)
                flat_normals = np.repeat(face_normals, 3, axis = 0).ravel().tolist()
                 
                flat_positions = verts_Ang[faces].ravel().tolist()
                num_mesh_vertices = faces.size
                flat_colors = color * num_mesh_vertices
                
                view.shape.add_mesh(flat_positions, flat_colors, None, flat_normals, "Isosurface")
                view.update_representation(component = len(view._ngl_component_names) - 1, repr_index = 0, side = "front", opacity = opacity, transparent = True, flatShading = False, depthWrite = True, opaqueBack = True)
        except:
            print("Problem creating the mesh")

        return view

    # Legacy function superseded by the LDOSGenerator
    def LDOS_maps(self, wavecar_object: vaspwfc, structure: Structure, energy_values_meV: list | np.ndarray = [], z_values_pm: list | np.ndarray | float | int = 0.,
                width_values_pm: list | np.ndarray | float | int = 0., gamma_meV: float = 50, n_gammas: int = 5, output_folder: str = "LDOS_maps"):
        # Initialize important parameters
        gamma_eV = gamma_meV / 1000
        gamma2 = gamma_eV ** 2
        voxels = wavecar_object._ngrid * 2
        voxel_size_Ang = np.diag(wavecar_object._Acell) / voxels
        px_per_pm = 1 / (100 * np.mean(voxel_size_Ang))
        atom_z_values_nm = structure.cart_coords[:, 2] * .1
        z_surface_nm = np.mean(np.partition(atom_z_values_nm, -12)[-12:-10])
        z_nm_per_vox = voxel_size_Ang[2] / 10
        n_spins = int(wavecar_object._nspin)
        n_kpts = int(wavecar_object._nkpts)
        
        # Get the band energies and take a selection ranging from n_gammas times the Lorentzian width below the minimum energy value to n_gammas times above the maximum energy value
        energy_dict = self.get_eigenenergies_from_wavecar(wavecar_object)
        spin_up_energies = energy_dict["energies"]["spin up"]
        spin_down_energies = energy_dict["energies"]["spin down"]
        k_resolved_spin_up_energies = spin_up_energies.reshape(n_kpts, -1)
        k_resolved_spin_down_energies = spin_down_energies.reshape(n_kpts, -1)        
        
        min_up_index = min([int(np.where(k_resolved_spin_up_energies[kpt] > .001 * np.min(energy_values_meV) - n_gammas * (gamma_meV / 1000))[0][0]) for kpt in range(len(k_resolved_spin_up_energies))])
        min_down_index = min([int(np.where(k_resolved_spin_down_energies[kpt] > .001 * np.min(energy_values_meV) - n_gammas * (gamma_meV / 1000))[0][0]) for kpt in range(len(k_resolved_spin_down_energies))])
        min_orbital_index = min((min_up_index, min_down_index))
        max_up_index = max([int(np.where(k_resolved_spin_up_energies[kpt] < .001 * np.max(energy_values_meV) + n_gammas * (gamma_meV / 1000))[0][-1]) for kpt in range(len(k_resolved_spin_up_energies))])
        max_down_index = max([int(np.where(k_resolved_spin_down_energies[kpt] < .001 * np.max(energy_values_meV) + n_gammas * (gamma_meV / 1000))[0][-1]) for kpt in range(len(k_resolved_spin_down_energies))])
        max_orbital_index = max((max_up_index, max_down_index))
        orbital_indices = np.arange(min_orbital_index, max_orbital_index + 1, 1, dtype = np.int32)

        selected_spin_up_energies = np.concatenate([k_resolved_spin_up_energies[kpt][orbital_indices] for kpt in range(n_kpts)])
        selected_spin_down_energies = np.concatenate([k_resolved_spin_down_energies[kpt][orbital_indices] for kpt in range(n_kpts)])
        energies = np.concatenate((selected_spin_up_energies, selected_spin_down_energies))
        
        # Extract a subset of the wavefunctions from the wavecar file and store it in wfns
        wfns = np.zeros((n_spins, n_kpts, len(orbital_indices), voxels[0], voxels[1], voxels[2]), dtype = np.complex64)
        for spin_index in range(n_spins):
            for k_index in range(n_kpts):
                for index, orb_index in enumerate(orbital_indices):
                    wfns[spin_index, k_index, index] = wavecar_object.wfc_r(spin_index + 1, k_index + 1, orb_index + 1)
        
        # Create the output directory and clean the tip height and width
        os.makedirs(os.path.join(os.curdir, output_folder), exist_ok = True)
        if isinstance(z_values_pm, int | float): z_values_pm = [z_values_pm]
        if not isinstance(z_values_pm, list | np.ndarray): print("Invalid height value(s)")
        if isinstance(width_values_pm, int | float): width_values_pm = [width_values_pm]
        if not isinstance(width_values_pm, list | np.ndarray): print("Invalid width value(s)")
        if isinstance(energy_values_meV, int | float): energy_values_meV = [energy_values_meV]
        if not isinstance(energy_values_meV, list | np.ndarray): print("Invalid energy value(s)")
        if not isinstance(energy_values_meV, list | np.ndarray) or not isinstance(width_values_pm, list | np.ndarray) or not isinstance(z_values_pm, list | np.ndarray): return

        # Loop over heights
        for z_slice_height_pm in z_values_pm:
            # Slice out the 2D wavefunction from the 3D wavefunction at the requested height
            z_target = z_surface_nm + z_slice_height_pm / 1000
            z_slice_index = int(z_target / z_nm_per_vox)
            wfns2D = wfns[:, :, :, :, :, z_slice_index]            
            spin_k_collapsed_wfns2D = wfns2D.reshape(-1, voxels[0], voxels[1]) # Flatten out the k and spin
            
            for width_pm in width_values_pm:
                # Broaden the wavefunction according to their overlap with the Gaussian tip wavefunction
                width_px = width_pm * px_per_pm
                broadened_wfns2D = [gaussian_filter(image, width_px, mode = "wrap") for image in spin_k_collapsed_wfns2D]
                densities = np.asarray(np.abs(np.array(broadened_wfns2D)) ** 2, dtype = np.float32)

                for target_energy_meV in energy_values_meV:
                    en_differences = np.array(energies, dtype = np.float32) - (.001 * target_energy_meV)
                    
                    weights = gamma_eV / (gamma2 + en_differences ** 2)
                    weights /= np.sum(weights)

                    image = np.average(densities, axis = 0, weights = weights)
                    plt.imsave(f"./{output_folder}/LDOS_w{int(round(width_pm))}pm_h{int(z_slice_height_pm)}pm@{int(round(target_energy_meV))}meV.png", image, cmap = "gray")                    
        return

    def LDOSGenerator(self, wavecar_object: vaspwfc, structure: Structure, energy_range_eV: list | np.ndarray = [], gamma_meV: float = 50, n_gammas: int = 5,
                      tip_width_pm: float = 0., tip_p_fraction: float = 0., tip_height_pm: float = 200.):
        return self.LDG(self, wavecar_object = wavecar_object, structure = structure, energy_range_eV = energy_range_eV, gamma_meV = gamma_meV, n_gammas = n_gammas,
                        tip_width_pm = tip_width_pm, tip_p_fraction = tip_p_fraction, tip_height_pm = tip_height_pm)

    class LDG:
        def __init__(self, initio_instance: Initio, wavecar_object: vaspwfc, structure: Structure, energy_range_eV: list | np.ndarray = [], gamma_meV: float = 50, n_gammas: int = 5,
                     tip_width_pm: float = 0., tip_p_fraction: float = 0., tip_height_pm = 200.):
            self.initio_instance = initio_instance
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
            energy_dict = self.initio_instance.get_eigenenergies_from_wavecar(wavecar_object)
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



        def set_tip_shape(self, width_pm: float = None, p_fraction: float = 0.) -> None:
            if isinstance(width_pm, float | int): self.tip_width_pm = width_pm
            if isinstance(p_fraction, float | int): self.tip_p_fraction = float(np.clip(p_fraction, 0, 1))
            return
        
        def set_tip_height(self, height_pm: float = None) -> None:
            if isinstance(height_pm, float | int): self.tip_height_pm = height_pm
            return
        
        def get_maps(self, energy_values_meV: float | int | list | np.ndarray = 0., height_values_pm: float | int | list | np.ndarray = None,
                     width_values_pm: float | int | list | np.ndarray = None, p_fractions: float | int | list | np.ndarray = None, output_folder: str = None) -> np.ndarray:
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

initio = Initio()

