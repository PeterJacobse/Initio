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
from typing import Literal



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
    return view

def orbital(wavecar_object: vaspwfc, ispin: int = 1, ikpt: int = 1, iband: int = 1, isolevel: float = .1, opacity: float = 1., flip_x: bool = False, flip_y: bool = False, flip_z: bool = False, upsampling: int = 1,
            struct: Structure | Molecule | None = None, max_bond_length: float = 2.6, atom_size: float = .3, bond_size: float = .22, struc_opacity: float = 1.,
            width: int = 800, height: int = 600, camera_type: str = "orthographic", flip_over: bool = False, background_color: str = "#000000") -> nv.NGLWidget:
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
        view.update_representation(component = len(view._ngl_component_names) - 2, repr_index = 0, opacity = struc_opacity, transparent = True, depthWrite = False)
        view.update_representation(component = len(view._ngl_component_names) - 1, repr_index = 0, opacity = struc_opacity, transparent = True, depthWrite = False)
    else:
        view = nv.NGLWidget()
    
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
    except:
        print("Problem creating the mesh")
    return view
