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
from scipy.ndimage import gaussian_filter



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

    def structure_plot(self, struc: Structure, max_bond_length: float = 2.6, width: int = 800, height: int = 600, atom_size: float = .3, bond_size: float = .22, camera_type: str = "orthographic") -> nv.NGLWidget:
        atoms = ase.AseAtomsAdaptor.get_atoms(struc)
        atoms.center()

        cutoffs = [max_bond_length / 2.0] * len(atoms)
        nl = NeighborList(cutoffs, skin = 0.0, bothways = True, self_interaction = False)
        nl.update(atoms)

        view = nv.show_ase(atoms)
        view.stage.set_parameters(depth_of_field = 0, fog_near = 100, fog_far = 100, camera_type = camera_type)
        
        bonds = [] #nv.shape.Shape(view = view)
        positions = atoms.get_positions()
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            for (neighbor_idx, offset) in zip(indices, offsets):
                if i < neighbor_idx: # Avoid creating duplicate overlapping cylinders
                    if np.any(offset != 0): continue 
                    
                    pos1 = positions[i].tolist()
                    pos2 = positions[neighbor_idx].tolist()
                    
                    bonds.append(["cylinder", pos1, pos2, [0.4, 0.4, 0.4], bond_size, "bond"])
                    #bonds.add_cylinder(pos1, pos2, [0.4, 0.4, 0.4], bond_size)
        view._add_shape(bonds)
        
        view.clear_representations()
        view.component_0.add_spacefill(radiusType = "vdw", radiusScale = atom_size)
        view.component_0.add_spacefill(selection = "_C", radiusType = "vdw", radiusScale = atom_size, colorValue = "#4D4D4D")
        view.component_0.add_spacefill(selection = "_N", radiusType = "vdw", radiusScale = atom_size + .1) # Emphasize nitrogen
        view.control.center(np.mean(struc.cart_coords, axis = 0))

        view.height = f"{height}px"
        view.width = f"{width}px"
        view.layout.height = f"{height}px"
        view.layout.width = f"{width}px"
        return view

    def orbital_plot(self, wavecar_object: vaspwfc, ispin: int = 1, ikpt: int = 1, iband: int = 1, isolevel: float = .1, opacity: float = 1., flip_x: bool = False, flip_y: bool = False, flip_z: bool = False,
                     struc: Structure = None, max_bond_length: float = 2.6, atom_size: float = .3, bond_size: float = .22, struc_opacity: float = 1., translate: list = [0, 0, 0],
                     width: int = 800, height: int = 600, camera_type: str = "orthographic") -> nv.NGLWidget:
        if not isinstance(wavecar_object, vaspwfc):
            print(f"Invalid wave function")
            return
        if not isinstance(opacity, float | int) or opacity < 0 or opacity > 1: opacity = 1.
        if not isinstance(struc_opacity, float | int) or struc_opacity < 0 or struc_opacity > 1: struc_opacity = 1.
        
        try:
            psi: np.ndarray = wavecar_object.wfc_r(ispin = ispin, ikpt = ikpt, iband = iband)
            if flip_x: psi = np.flip(psi, axis = 0)
            if flip_y: psi = np.flip(psi, axis = 1)
            if flip_z: psi = np.flip(psi, axis = 2)
            orb_plus = np.abs(np.clip(psi, a_min = 0, a_max = np.inf)) ** 2
            orb_minus = np.abs(np.clip(psi, a_min = -np.inf, a_max = 0) ** 2)
        
            cell_size_Ang = wavecar_object._Acell
            voxels = wavecar_object._ngrid * 2
            
            voxel_size = np.diag(cell_size_Ang) / voxels
        except Exception as e:
            print(f"{e}")
            return
        
        if isinstance(struc, Structure):
            view = self.structure_plot(struc, max_bond_length, width, height, atom_size, bond_size, camera_type)
            view.update_representation(component = len(view._ngl_component_names) - 2, repr_index = 0, opacity = struc_opacity, transparent = True, depthWrite = False)
            view.update_representation(component = len(view._ngl_component_names) - 1, repr_index = 0, opacity = struc_opacity, transparent = True, depthWrite = False)
        else:
            view = nv.NGLWidget()
        
        try:
            for orb, color in zip([orb_plus, orb_minus], [[.8, .4, 0], [0, .2, .9]]):
                (verts, faces, normals, values) = measure.marching_cubes(orb, level = isolevel * np.max(orb_plus))
                verts_Ang = verts * voxel_size + translate
                
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
                
                view.shape.add_mesh(flat_positions, flat_colors, None, flat_normals, "Isosurface Mesh")
                view.update_representation(component = len(view._ngl_component_names) - 1, repr_index = 0, opacity = opacity, transparent = True, depthWrite = False)
        except:
            print("Problem creating the mesh")

        return view

    def LDOS_maps(self, wavecar_object: vaspwfc, struc: Structure, energy_values_meV: list | np.ndarray = [], z_values_pm: list | np.ndarray | float | int = 0.,
                width_values_pm: list | np.ndarray | float | int = 0., gamma_meV: float = 50, n_gammas: int = 5, output_folder: str = "LDOS_maps"):
        # Initialize important parameters
        gamma_eV = gamma_meV / 1000
        gamma2 = gamma_eV ** 2
        voxels = wavecar_object._ngrid * 2
        voxel_size_Ang = np.diag(wavecar_object._Acell) / voxels
        px_per_pm = 1 / (100 * np.mean(voxel_size_Ang))
        atom_z_values_nm = struc.cart_coords[:, 2] * .1
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


