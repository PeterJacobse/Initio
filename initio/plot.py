import numpy as np
import matplotlib.colors as cols
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from .structures import Structure, Molecule
from .VaspBandUnfolding import vaspwfc
import nglview as nv
from ase.neighborlist import NeighborList
from pymatgen.io.ase import AseAtomsAdaptor
from ase.data.colors import jmol_colors
from scipy.ndimage import zoom
from skimage.measure import marching_cubes
from .bands import spin_and_occupation_resolved_DOS
from .math import complex_to_rgba
from typing import Literal
import plotly.graph_objects as go



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



def complex_array(array: np.ndarray, phase_shift: float = 0.) -> plt.Figure:
    """Convenience function for plotting a complex array

    Args:
        array (np.ndarray): Numpy array
        phase_shift (float, optional): Phase shift. Defaults to 0..
    """
    rgba_array = complex_to_rgba(array, phase_shift = phase_shift)
    fig, ax = plt.subplots()
    ax.imshow(rgba_array)
    return fig

def DOS(wavecar_object: vaspwfc, *args, **kwargs) -> plt.Figure:
    colors = kwargs.pop("colors", None)
    
    # No colors given. Use defaults
    if not isinstance(colors, list) or len(colors) < 2: colors = ["#A00000", "#0000A0"]
    # Invalid colors given. Use defaults
    if not cols.is_color_like(colors[0]): colors = ["#A00000", "#0000A0"]

    col_up_occ = list(cols.to_rgb(colors[0]))
    col_up_unocc = [.5 + .5 * channel for channel in col_up_occ]
    col_down_occ = list(cols.to_rgb(colors[1]))
    col_down_unocc = [.5 + .5 * channel for channel in col_down_occ]
    
    LDOS_up_occ, LDOS_up_unocc, LDOS_down_occ, LDOS_down_unocc = spin_and_occupation_resolved_DOS(wavecar_object, *args, **kwargs)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(3, 4.6)
    ax.fill_betweenx(LDOS_up_occ[0], LDOS_up_occ[1], color = col_up_occ)
    ax.fill_betweenx(LDOS_up_unocc[0], LDOS_up_unocc[1], color = col_up_unocc)
    ax.fill_betweenx(LDOS_down_occ[0], -LDOS_down_occ[1], color = col_down_occ)
    ax.fill_betweenx(LDOS_down_unocc[0], -LDOS_down_unocc[1], color = col_down_unocc)
    
    ax.set_xlabel("DOS up (a.u.)    DOS down (a.u.)")
    ax.set_ylabel("energy (eV)")
    ax.set_xticks([])
    
    en_range = kwargs.get("energy_range", [0, 1])
    ax.set_ylim(en_range[0], en_range[1])
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(.1))
    
    ax.grid(True, which = "both", axis = "y", color = "gray", linewidth = 0.5, alpha = 0.5)
    return fig

def levels(wavecar_object: vaspwfc | None = None, energies: list | np.ndarray = [], occupations: list | np.ndarray = [], spins: list | np.ndarray = [], band_indices: list | np.ndarray = [], kpoints: list | np.ndarray = [],
           en_diff_min = 0.01, spin_up_color: str = "#3080ff", spin_down_color: str = "#d00050", figsize: tuple = (750, 600), energy_range: list | tuple | np.ndarray | None = None, channel_gap: float = 1.5) -> go.Figure:

    if isinstance(wavecar_object, vaspwfc):
        n_spins = wavecar_object._nspin
        n_kpoints = wavecar_object._nkpts

        all_energies_all_k = wavecar_object._bands
        all_energies = all_energies_all_k[:, 0]
        all_band_indices = np.broadcast_to(np.arange(all_energies.shape[-1]), all_energies.shape)
        all_occupations_all_k = wavecar_object._occs
        all_occupations = all_occupations_all_k[:, 0]
        all_spins = np.zeros_like(all_energies, dtype = np.int8)

        if n_spins > 1:
            all_spins[1] += 1
        #if n_kpoints > 1:
        #    kvectors = np.round(wfc._kvecs, 3)
        #    all_kvectors = np.broadcast_to(np.arange(all_energies.shape[-1]), all_energies.shape)

        if energy_range is not None and len(energy_range) == 2: en_min, en_max = float(energy_range[0]), float(energy_range[1])
        else: en_min, en_max = min(all_energies), max(all_energies)
        
        band_crosses_min = np.any(all_energies_all_k > en_min, axis = (0, 1))
        band_min = int(np.where(band_crosses_min)[0][0])
        band_crosses_max = np.any(all_energies_all_k < en_max, axis = (0, 1))
        band_max = int(np.where(band_crosses_max)[0][-1]) + 2

        #cropped_kpoints = all_kpoints[:, :, band_min:band_max]
        cropped_energies = all_energies[:, band_min:band_max]
        cropped_occupations = all_occupations[:, band_min:band_max]
        cropped_spins = all_spins[:, band_min:band_max]
        cropped_band_indices = all_band_indices[:, band_min:band_max]

        #kpoints_flat = cropped_kpoints.reshape(-1)
        energies = cropped_energies.reshape(-1)
        occupations = cropped_occupations.reshape(-1)
        spins = cropped_spins.reshape(-1)
        band_indices = cropped_band_indices.reshape(-1)
    
    energies = np.asarray(energies, dtype = np.float32)
    has_spin = True if (isinstance(spins, np.ndarray | list) and len(spins) == len(energies)) else False
    has_indices = True if (isinstance(band_indices, list | np.ndarray) and len(band_indices) == len(energies)) else False
    has_occupations = True if (isinstance(occupations, list | np.ndarray) and len(occupations) == len(energies)) else False
    has_kpoints = True if (isinstance(kpoints, list | np.ndarray) and len(kpoints) == len(energies)) else False
    
    # Cropping to energy range
    if energy_range is not None and len(energy_range) == 2: en_min, en_max = float(energy_range[0]), float(energy_range[1])
    else: en_min, en_max = min(all_energies), max(all_energies)
    
    band_min = np.where(energies > en_min)[0]
    band_min = band_min[0] if len(band_min) > 0 else 0
    band_max = np.where(energies < en_max)[0]
    band_max = band_max[-1] + 2 if len(band_max) > 0 else len(energies)

    energies = energies[band_min:band_max]
    if has_spin: spins = spins[band_min:band_max]
    if has_occupations: occupations = occupations[band_min:band_max]
    if has_kpoints: kpoints = kpoints[band_min:band_max]
    if has_indices: band_indices = band_indices[band_min:band_max]
    else: band_indices = np.arange(len(energies))


    
    # Level clustering and horizontal partitioning
    def get_centered_sub_slots(indices_in_channel: list[int] | np.ndarray) -> dict[int, float]:
        if len(indices_in_channel) == 0: return {}
        
        channel_energies = energies[indices_in_channel]
        sorted_sub_idx = np.argsort(channel_energies)
        
        # 1. Group levels into clusters that overlap vertically
        clusters = []
        current_cluster = []
        last_energy = -float("inf")
        
        for sub_idx in sorted_sub_idx:
            en = channel_energies[sub_idx]
            if en < en_min or en > en_max: continue
            global_idx = indices_in_channel[sub_idx]

            if en - last_energy >= en_diff_min:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [global_idx]
            else:
                current_cluster.append(global_idx)
            last_energy = en
        if current_cluster: clusters.append(current_cluster)

        slots_mapped = {}
        for cluster in clusters:
            length = len(cluster)
            positions = np.arange(length) - .5 * (length - 1)
            for global_idx, pos in zip(cluster, positions): slots_mapped.update({int(global_idx): float(pos)})
        return slots_mapped
    
    sub_slot_spacing = 0.5
    level_width = 0.4
    fig = go.Figure()



    if has_spin:        
        down_indices = [i for i, s in enumerate(spins) if s == 0]
        down_slots = get_centered_sub_slots(down_indices)    
        up_indices = [i for i, s in enumerate(spins) if s == 1]
        up_slots = get_centered_sub_slots(up_indices)
        max_sub_offset = max([abs(value) for value in list(down_slots.values()) + list(up_slots.values())]) * sub_slot_spacing
    else:
        up_indices = np.arange(len(energies))
        up_slots = get_centered_sub_slots(up_indices)
        max_sub_offset = max([abs(value) for value in up_slots.values()]) * sub_slot_spacing

    for index, energy in enumerate(energies):
        if energy < en_min or energy > en_max: continue
        
        band_index = band_indices[index]
        tooltip = f"<b>Band / Level index:</b> {band_index}<br><b>energy:</b> {energy:.3f} eV"
        
        if has_spin:
            match spins[index]:
                case 1:
                    channel_center = .5 * channel_gap
                    x_center = channel_center + up_slots[index] * sub_slot_spacing
                    color = spin_up_color
                    spin_label = "↑"
                case _:
                    channel_center = -.5 * channel_gap
                    x_center = channel_center - (down_slots[index] * sub_slot_spacing)
                    color = spin_down_color
                    spin_label = "↓"
            tooltip += f"<br><b>spin:</b> {spin_label}"
        else:
            spin_label = "↑↓"
            x_center = up_slots[index] * sub_slot_spacing
            color = spin_down_color
        
        if has_occupations:
            occ = occupations[index]
            tooltip += f"<br><b>occupation:</b> {occ:.2f}<br><extra></extra>"
            if occ > 0: fig.add_annotation(x = x_center, y = energy, text = spin_label, showarrow = False, font = {"size": int(32 * occ), "color": "#ffffff"}, yshift = 0)
        
        if has_kpoints:
            kpoint_index = kpoints[index]
            tooltip += f"<br><b>k-point:</b> {kpoint_index:.2f}<br><extra></extra>"

        fig.add_trace(go.Scatter(x = [x_center - .5 * level_width, x_center, x_center + .5 * level_width], y = [energy, energy, energy], mode = "lines", line = {"width": 4, "color": color}, hovertemplate = tooltip))

    x_range_max = max_sub_offset + .6
    if has_spin: x_range_max += .5 * channel_gap
    
    x_settings = {"tickvals": [-channel_gap / 2, channel_gap / 2], "ticktext": ["<b>spin down (↓)</b>", "<b>spin up (↑)</b>"], "zeroline": False, "range": [-x_range_max, x_range_max], "showgrid": False, "showticklabels": True}
    y_settings = {"title": "Energy (eV)", "showgrid": True, "gridcolor": "rgba(220, 220, 220, 0.5)", "range": [en_min, en_max]}
    if has_spin: x_settings.update({"zeroline": True, "zerolinecolor": "rgba(180, 180, 180, 0.4)", "zerolinewidth": 1.5, })
    fig.update_layout(title = {"text": "energy level diagram", "x": 0.5}, showlegend = False, hovermode = "closest", width = figsize[0], height = figsize[1], template = "plotly_dark", xaxis = x_settings, yaxis = y_settings)
    return fig

def structure(struct: Structure | Molecule, max_bond_length: float | None = None, width: int = 800, height: int = 600, atom_size: float = .3, bond_size: float = .22, camera_type: Literal["orthographic", "perspective"] = "orthographic", flip_over: bool = False, background_color: str = "#000000") -> nv.NGLWidget:
    """
    Generates a NGLView view object that contains the structure.

    Args:
        structure (Structure, Molecule, optional): Structure. Defaults to None.
        max_bond_length (float, optional): Max distance between atoms that is still interpreted as a bond. Defaults to 2.6.
        atom_size (float, optional): Atom size relative to its vdW radius. Defaults to .3.
        bond_size (float, optional): Bond size. Defaults to .22.
        width (int, optional): Number of pixels of the outputted view. Defaults to 800.
        height (int, optional): Number of lines of the outputted view. Defaults to 600.
        camera_type (str, optional): "orthographic" or "perspective". Defaults to "orthographic".
        flip_over (bool, optional): Whether to flip the camera perspective to below the xy plane. Defaults to False.
        background_color (str, optional): Background color. Defaults to "#000000".

    Returns:
        nv.NGLWidget: NGLView view object
    """
    atoms = AseAtomsAdaptor.get_atoms(struct)
    
    Z = list(struct.atomic_numbers)
    R = struct.cart_coords
    Zcolors = np.array([.5 * rgb if atomic_number > 0 else (0, 0, 0) for atomic_number, rgb in enumerate(jmol_colors)])
    Zcolors[6] = [.1, .1, .1]
    
    if not max_bond_length:
        if 74 in Z: max_bond_length = 2.6 # Shortcut for working with TMDs
        else: max_bond_length = 1.6 # Shortcut fallback for organic stuff

    cutoffs = [max_bond_length / 2.0] * len(atoms) # type: ignore
    nl = NeighborList(cutoffs, skin = 0.0, bothways = True, self_interaction = False)
    nl.update(atoms)
    
    view: nv.NGLWidget = nv.show_ase(atoms)
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
    view.control.center(np.mean(struct.cart_coords, axis = 0))
    if flip_over: view.control.spin([1, 0, 0], np.deg2rad(180))
    view.center()

    view.height = f"{height}px"
    view.width = f"{width}px"
    view.layout.height = f"{height}px"
    view.layout.width = f"{width}px"
    view.layout.background = background_color
    view.layout.border = "none"
    view.layout.margin = "0"
    view.layout.padding = "0"
    return view

def orbital(wavecar_object: vaspwfc, spin: int | str = 0, kpoint: int = 0, band: int = 0, isolevel: float = .1, opacity: float = 1., flip_x: bool = False, flip_y: bool = False, flip_z: bool = False, upsampling: int = 1,
            struct: Structure | Molecule | None = None, max_bond_length: float = 2.6, atom_size: float = .3, bond_size: float = .22, struc_opacity: float = 1.,
            width: int = 800, height: int = 600, camera_type: Literal["orthographic", "perspective"] = "orthographic", flip_over: bool = False, background_color: str = "#000000") -> nv.NGLWidget:
    """
    Generates a NGLView view object that contains the structure along with a 3D orbital contour plot of the single-particle state defined by the provided wavecar object, ispin, ikpt and iband.

    Args:
        wavecar_object (vaspwfc): Wavecar object from pyvaspwfc
        ispin (int, optional): Spin index. Defaults to 1.
        ikpt (int, optional): K-point index. Defaults to 1.
        iband (int, optional): Band index. Defaults to 1.
        isolevel (float, optional): Cut-off value of psi^2 where 3D contours are generated. Defaults to .1.
        opacity (float, optional): Opacity of the orbitals. Defaults to 1..
        flip_x (bool, optional): Bool to flip the wavefunction in the yz plane relative to the structure. Should not generally be used. Defaults to False.
        flip_y (bool, optional): Bool to flip the wavefunction in the xz plane relative to the structure. Should not generally be used. Defaults to False.
        flip_z (bool, optional): Bool to flip the wavefunction in the xy plane relative to the structure. Should not generally be used. Defaults to False.
        upsampling (int, optional): Value to resample the cube data (numpy array) to before plotting the orbital. Higher values are more expensive but generate smoother orbitals. Defaults to 1 (output array size = 1 * input array size).
        structure (Structure, Molecule, optional): Structure. Defaults to None.
        max_bond_length (float, optional): Max distance between atoms that is still interpreted as a bond. Defaults to 2.6.
        atom_size (float, optional): Atom size relative to its vdW radius. Defaults to .3.
        bond_size (float, optional): Bond size. Defaults to .22.
        struc_opacity (float, optional): Opacity of the structure. Defaults to 1..
        width (int, optional): Number of pixels of the outputted view. Defaults to 800.
        height (int, optional): Number of lines of the outputted view. Defaults to 600.
        camera_type (str, optional): "orthographic" or "perspective". Defaults to "orthographic".
        flip_over (bool, optional): Whether to flip the camera perspective to below the xy plane. Defaults to False.
        background_color (str, optional): Background color. Defaults to "#000000".

    Returns:
        nv.NGLWidget: NGLView view object
    """
    if isinstance(spin, str):
        if not spin.lower() in {"down", "up"}: raise Exception(f"Invalid spin {spin}. Recognized values are \"up\" and \"down\".")
        else: spin = 0 if spin.lower() == "down" else 1

    ispin = spin + 1
    ikpt = kpoint + 1
    iband = band + 1
    
    if not isinstance(wavecar_object, vaspwfc): raise Exception(f"Invalid wave function")
    if not isinstance(opacity, float | int) or opacity < 0 or opacity > 1: opacity = 1.
    if not isinstance(struc_opacity, float | int) or struc_opacity < 0 or struc_opacity > 1: struc_opacity = 1.
    
    try:
        bare_psi: np.ndarray = wavecar_object.wfc_r(ispin = ispin, ikpt = ikpt, iband = iband) # type: ignore
        psi: np.ndarray = zoom(bare_psi, zoom = upsampling, order = 3) # type: ignore
        if flip_x: psi = np.flip(psi, axis = 0)
        if flip_y: psi = np.flip(psi, axis = 1)
        if flip_z: psi = np.flip(psi, axis = 2)
        orb_plus = np.abs(np.clip(psi, a_min = 0, a_max = np.inf)) ** 2
        orb_minus = np.abs(np.clip(psi, a_min = -np.inf, a_max = 0) ** 2)
    
        cell_size_Ang = wavecar_object._Acell
        voxels = wavecar_object._ngrid * 2 * upsampling
        
        voxel_size = np.diag(cell_size_Ang) / voxels
    except Exception as e:
        raise Exception(f"Problem creating the orbital plot: {e}")
    
    if isinstance(struct, Structure):
        view = structure(struct, max_bond_length, width, height, atom_size, bond_size, camera_type, background_color = background_color)
        view.update_representation(component = len(view._ngl_component_names) - 2, repr_index = 0, opacity = struc_opacity, transparent = True, depthWrite = True)
        view.update_representation(component = len(view._ngl_component_names) - 1, repr_index = 0, opacity = struc_opacity, transparent = True, depthWrite = True)
    else:
        view = nv.NGLWidget()
        view.stage.set_parameters(depth_of_field = 0, fog_near = 100, fog_far = 100, camera_type = camera_type, background_color = background_color)
    
    try:
        for orb, color in zip([orb_plus, orb_minus], [[.8, .4, 0], [0, .2, .9]]):
            verts, faces, normals, values = marching_cubes(orb, level = isolevel * np.max(orb_plus))
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
            
            view.shape.add_mesh(flat_positions, flat_colors, None, flat_normals, "Isosurface") # type: ignore
            view.update_representation(component = len(view._ngl_component_names) - 1, repr_index = 0, side = "front", opacity = opacity, transparent = True, flatShading = False, depthWrite = True, opaqueBack = True)
    
        if flip_over: view.control.spin([1, 0, 0], np.deg2rad(180))
        view.center()
        view.layout.background = background_color
        view.layout.border = "none"
        view.layout.margin = "0"
        view.layout.padding = "0"
    except:
        print("Problem creating the mesh")
    return view
