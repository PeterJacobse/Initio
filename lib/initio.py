import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors as cols
import numpy as np
from . import vaspwfc
import pymatgen.core.structure as pmg_struct
from pymatgen.core import periodic_table
from pymatgen.io import vasp, ase
import nglview as nv
from skimage import measure
from ase.neighborlist import NeighborList
from ase.data.colors import jmol_colors
from scipy.ndimage import gaussian_filter, zoom, sobel



class Initio:
    def __init__(self):
        cm = nv.color.ColormakerRegistry
        cm.add_scheme_func('custom_carbon', '''
            this.atomColor = function (atom) {
                if (atom.element == "C") {
                    return 0x333333; // Hex code format for JavaScript
                } else {
                    return 0xcccccc; // Default color for other atoms
                }
            }
        ''')



    def read_vasp_file(self, path: str) -> vaspwfc | Initio.Structure | bool:
        base_name = os.path.basename(path)
        
        match base_name:
            case "WAVECAR": return self.get_wavecar(path)
            case "POSCAR" | "CONTCAR": return self.get_structure(path)
            case "PROCAR": return self.get_procar(path)
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

    def get_structure(self, path: str) -> Initio.Structure:
        try:
            structure = self.Structure.from_file(path)
            return structure
        except Exception as e:
            print(f"Error loading the structure: {e}")
            return False

    def get_procar(self, path: str) -> vasp.Procar:
        procar = None
        try: procar = vasp.outputs.Procar(path)
        except Exception as e: print(f"Could not open PROCAR file: {e}")
        return procar

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

    def structure_plot(self, structure: Initio.Structure, max_bond_length: float = None, width: int = 800, height: int = 600, atom_size: float = .3, bond_size: float = .22,
                       camera_type: str = "orthographic", flip_over: bool = False, background_color: str = "#000000") -> nv.NGLWidget:
        atoms = ase.AseAtomsAdaptor.get_atoms(structure)
        Z = list(structure.atomic_numbers)
        R = structure.cart_coords
        Zcolors = np.array([.5 * rgb if atomic_number > 0 else (0, 0, 0) for atomic_number, rgb in enumerate(jmol_colors)])
        Zcolors[6] = [.1, .1, .1]
        
        if not max_bond_length:
            if 74 in Z: max_bond_length = 2.6 # Shortcut for working with TMDs
            else: max_bond_length = 1.6 # Shortcut fallback for organic stuff

        cutoffs = [max_bond_length / 2.0] * len(atoms)
        nl = NeighborList(cutoffs, skin = 0.0, bothways = True, self_interaction = False)
        nl.update(atoms)
        
        view = nv.show_ase(atoms)
        view.stage.set_parameters(depth_of_field = 0, fog_near = 100, fog_far = 100, camera_type = camera_type, background_color = background_color)
        view._execute_js_code("""
            var stage = this.stage;

            // 1. Turn off nglview's built-in mouseover tooltip text engine
            stage.setParameters({ tooltip: false });

            // 2. Create or grab our isolated HTML tooltip element
            var customTooltip = document.getElementById("custom-ngl-tooltip");
            if (!customTooltip) {
                customTooltip = document.createElement("div");
                customTooltip.id = "custom-ngl-tooltip";
                customTooltip.style.position = "fixed";  // Fixed positioning prevents scroll/canvas coordinate offset errors
                customTooltip.style.zIndex = "10005";
                customTooltip.style.background = "rgba(0, 0, 0, 0.85)";
                customTooltip.style.color = "white";
                customTooltip.style.padding = "4px 8px";
                customTooltip.style.borderRadius = "4px";
                customTooltip.style.fontFamily = "monospace";
                customTooltip.style.fontSize = "12px";
                customTooltip.style.pointerEvents = "none";
                customTooltip.style.display = "none";
                document.body.appendChild(customTooltip);
            }

            // 3. Track actual mouse screen coordinates on the container
            var mouseX = 0;
            var mouseY = 0;
            stage.viewer.container.addEventListener('mousemove', function(e) {
                mouseX = e.clientX;
                mouseY = e.clientY;
                
                // If tooltip is visible, update its position dynamically with the mouse movement
                if (customTooltip.style.display === "block") {
                    customTooltip.style.left = (mouseX + 15) + "px";
                    customTooltip.style.top = (mouseY + 15) + "px";
                }
            });

            // 4. Update the content when an atom is hovered
            stage.signals.hovered.add(function(pickingProxy) {
                if (pickingProxy && pickingProxy.atom) {
                    var atom = pickingProxy.atom;
                    
                    // Build the clean string showing name and index
                    customTooltip.innerText = atom.qualifiedName() + " (Index: " + atom.index + ")";
                    
                    // Position near the recorded mouse coordinates and display
                    customTooltip.style.left = (mouseX + 15) + "px";
                    customTooltip.style.top = (mouseY + 15) + "px";
                    customTooltip.style.display = "block";
                } else {
                    customTooltip.style.display = "none";
                }
            });
        """)
        
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
        view.component_0.add_spacefill(selection = "_C", radiusType = "vdw", radiusScale = atom_size, color = "custom_carbon")
        view.component_0.add_spacefill(selection = "_N", radiusType = "vdw", radiusScale = atom_size + .1, colorValue = 2 * Zcolors[8]) # Emphasize nitrogen
        view.control.center(np.mean(structure.cart_coords, axis = 0))
        if flip_over: view.control.spin([1, 0, 0], np.deg2rad(180))
        view.center()

        view.height = f"{height}px"
        view.width = f"{width}px"
        view.layout.height = f"{height}px"
        view.layout.width = f"{width}px"
        view.layout.background = background_color
        return view

    def orbital_plot(self, wavecar_object: vaspwfc, ispin: int = 1, ikpt: int = 1, iband: int = 1, isolevel: float = .1, opacity: float = 1., flip_x: bool = False, flip_y: bool = False, flip_z: bool = False, upsampling: int = 1,
                     structure: Structure = None, max_bond_length: float = 2.6, atom_size: float = .3, bond_size: float = .22, struc_opacity: float = 1.,
                     width: int = 800, height: int = 600, camera_type: str = "orthographic", flip_over: bool = False, background_color: str = "#000000") -> nv.NGLWidget:
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
        
        if isinstance(structure, Initio.Structure):
            view = self.structure_plot(structure, max_bond_length, width, height, atom_size, bond_size, camera_type, background_color = background_color)
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
        
            if flip_over: view.control.spin([1, 0, 0], np.deg2rad(180))
            view.center()
            view.layout.background = background_color
        except:
            print("Problem creating the mesh")

        return view

    class LDOSGenerator:
        def __init__(self, wavecar_object: vaspwfc, structure: Initio.Structure, energy_range_eV: list | np.ndarray = [], gamma_meV: float = 50, n_gammas: int = 5,
                     tip_width_pm: float = 0., tip_p_fraction: float = 0., tip_height_pm = 200.):
            initio_instance = Initio()
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
            energy_dict = initio_instance.get_eigenenergies_from_wavecar(wavecar_object)
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

    class Structure(pmg_struct.Structure):
        # Subclass of the pmg.core.Structure class with convenience function exchange_atom and a graphene nanoribbon constructor added
        def exchange_atom(self, index: int, element: str | int) -> None:
            if isinstance(element, int): # Convert from atomic number to element symbol
                elements = {el.Z: el.symbol for el in periodic_table.Element}
                element = elements[element]
            if not isinstance(index, int) or not isinstance(element, str): return
            
            self[index].species = element
            return

        @classmethod
        def GNR(cls, N: int = 2, orientation: str = "armchair", n_supercell: int = 1, unit_cell_height_A: float | int = 10) -> Initio.Structure:
            match orientation.lower():
                case "a": orientation = "a"
                case "armchair": orientation = "a"
                case "z": orientation = "z"
                case "zigzag": orientation = "z"
                case _:
                    print("Invalid orientaion")
                    return
            
            latvec_z = unit_cell_height_A
            xlist = np.zeros((N * 2), dtype = np.float32)
            ylist = np.zeros((N * 2), dtype = np.float32)
            atomlist = np.full((N * 2), "C", dtype = np.str_)
            CC_length = 1.42
            CH_length = 1.09
            
            match orientation:
                case "a":
                    row_to_row = .25 * CC_length * np.sqrt(3) # Row-to-row spacing in the y direction
                    
                    for i in range(N):
                        xdist_from_axis = .5 * CC_length * (np.mod(i, 2) + 1)
                        sign = 1 - 2 * np.mod(i, 2)
                        xlist[i] = sign * xdist_from_axis
                        xlist[i + N] = -sign * xdist_from_axis
                        ylist[i] = 2 * i * row_to_row
                        ylist[i + N] = 2 * i * row_to_row

                    # Pad 2 hydrogen atoms to the bottom and 2 to the top of the GNR
                    xlist = np.append(xlist, [.5 * CC_length + .5 * CH_length, -.5 * CC_length - .5 * CH_length,
                                              -CC_length * (.25 * np.mod((N + 1) * 2, 4) + .5) + CH_length * (.5 - np.mod(N, 2)), CC_length * (.25 * np.mod((N + 1) * 2, 4) + .5) - CH_length * (.5 - np.mod(N, 2))])
                    ylist = np.append(ylist, [-CH_length * np.sqrt(3) / 2, -CH_length * np.sqrt(3) / 2,
                                              max(ylist) + CH_length * np.sqrt(3) / 2, max(ylist) + CH_length * np.sqrt(3) / 2])
                    atomlist = np.append(atomlist, ["H", "H", "H", "H"])
                    
                    latvec_x = 3 * CC_length # Simply the unit cell length
                    latvec_y = (N + 1) * 4 * row_to_row # The GNR hard wall boundary conditions dictate that the transverse component of the wavefunctions have nodal planes on atomic row 0 and atomic row N + 1
                    # Thus, using this width ensures that the unit cell exactly fits the wavelength of the transverse waves and integer multiples of it
                    # This prevents having to represent the transverse nodal plane structure with lots of different waves, also known as Fourier leakage
                
                case "z":
                    row_to_row = 1.5 * CC_length # Row-to-row spacing in the y direction
                    
                    for i in range(0, N * 2, 2):
                        xlist[i] = -.25 * np.sqrt(3) * CC_length * (np.mod(i, 4) - 1) # Atom to the left side of the x axis
                        xlist[i + 1] = .25 * np.sqrt(3) * CC_length * (np.mod(i, 4) - 1) # Atom to the right side of the x axis
                        ylist[i] = .5 * i * row_to_row - .25 * CC_length
                        ylist[i + 1] = .5 * i * row_to_row + .25 * CC_length
                    
                    # Pad 1 hydrogen atom to the bottom and 1 to the top of the GNR
                    xlist = np.append(xlist, [xlist[0], xlist[-1]])
                    ylist = np.append(ylist, [-.25 * CC_length - CH_length, ylist[-1] + CH_length])
                    atomlist = np.append(atomlist, ["H", "H"])
                    
                    latvec_x = np.sqrt(3) * CC_length # Simply the unit cell length
                    latvec_y = (N + 1) * 2 * row_to_row # The GNR hard wall boundary conditions dictate that the transverse component of the wavefunctions have nodal planes on atomic row 0 and atomic row N + 1
                    # Thus, using this width ensures that the unit cell exactly fits the wavelength of the transverse waves and integer multiples of it
                    # This prevents having to represent the transverse nodal plane structure with lots of different waves, also known as Fourier leakage

            ylist -= np.mean(ylist) # Shift the y coordinates so that the structure is centered around the origin
            xlist_shifted = xlist + .5 * latvec_x # Move the atoms from being centered around the origin to being centered around the center of the unit cell
            ylist_shifted = ylist + .5 * latvec_y
            
            coords = np.array([xlist_shifted, ylist_shifted, np.zeros_like(xlist) + .5 * latvec_z]).T            
            structure = Initio.Structure(lattice = pmg_struct.Lattice.from_parameters(latvec_x, latvec_y, latvec_z, 90, 90, 90), species = atomlist,
                                         coords = coords, coords_are_cartesian = True)
            if isinstance(n_supercell, int): structure.make_supercell([n_supercell, 1, 1])
            return structure
