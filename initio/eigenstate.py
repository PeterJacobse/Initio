import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from .VaspBandUnfolding import vaspwfc
from .math import complex_to_rgba
from io import BytesIO
import base64
from IPython import display
from typing import Literal



class Eigenstate:
    """Eigenstate object, comprising both the single-particle wavefunction and the single-particle energy (eigenenergy).
    Unlike the pyvaspwfc convention of 1-indexing (e.g. ispin = 1 or 2), Eigenstate uses 0-based indexing like Python.
    """
    def __init__(self, energy: float, spin: int | str | bool = 0, band: int = 0, psi: np.ndarray = np.zeros((3, 3, 3)), latvecs: np.ndarray = np.zeros((3, 3), dtype = np.float32), kvec_reciprocal: np.ndarray = np.zeros((3), dtype = np.float32),
                 kvec_cartesian: np.ndarray = np.zeros((3), dtype = np.float32), kpoint: int = 0, slice_axis: Literal[0, 1, 2] = 2):
        if not psi.ndim == 3: raise Exception(f"Invalid dimensionality {psi.ndim} for eigenstate wavefunction")
        self.clean_spin(spin) # Creates attributes self.spin and self.spin_name
        self.energy = energy
        self.eigenenergy = energy # Alias
        self.latvecs = latvecs
        self.kvec_reciprocal = kvec_reciprocal
        self.kvec_cartesian = kvec_cartesian
        self.kpoint = kpoint
        self.band = band
        self.psi = psi
        self.slice_axis = slice_axis # Default axis to slice the 3D wavefunction object along when extracting a 2D slice
   
    def __repr__(self) -> str:
        return self.info()

    def _repr_mimebundle_(self, include = None, exclude = None):
        """
        IPython hook to automatically display a rich media thumbnail inline inside Jupyter Notebooks and VS Code.
        """
        repr_text = self.info()
        try:
            image_bytes = self.generate_thumbnail()
            b64_string = base64.b64encode(image_bytes).decode('utf-8')
            html_content = f"""
            <div style="display: flex; flex-direction: column; align-items: flex-start; gap: 8px;">
                <div style="white-space: pre-wrap; div style="font-family: monospace; color: var(--vscode-editor-foreground, inherit);">{repr_text}</div>
                <img src="data:image/png;base64,{b64_string}" style="max-width: 200px; height: auto;" />
            </div>
            """
            return {'text/html': html_content, 'text/plain': repr_text}
        except Exception as e:
            return {'text/plain': repr_text + f" (Thumbnail failed: {e})>"}

    def info(self) -> str:
        result1 = f"Eigenstate object\n  Energy:\t\t{self.eigenenergy = } eV\n  Spin:\t\t\t{self.spin = }\n\t\t\t{self.spin_name = }\n  K-vector:\t\t{self.kvec_cartesian = } * (2 pi / A)"
        result2 = f"\n  Band/orbital index:\t{self.band = }\n  Lattice vectors:\tself.latvecs = array([{[float(value) for value in self.latvecs[0]]}], [{[float(value) for value in self.latvecs[1]]}], [{[float(value) for value in self.latvecs[2]]}])"
        result3 = f"\n  Wavefunction:\t\tself.psi = <np.{self.psi.__class__.__name__} (shape = {self.psi.shape})>"
        return result1 + result2 + result3

    def set_slice_axis(self, axis: Literal[-1, 0, 1, 2] = -1) -> None:
        if axis in {0, 1, 2}: self.slice_axis = axis
        return

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

    def slice_wavefunction(self, index: int | bool = False, axis: Literal[-1, 0, 1, 2] = -1) -> None:
        """Generates 2D numpy arrays representing slices of the wavefunction. This function saves the wavefunction itself and its p-wave tip form to attributes self.psi2D_s and self.psi2D_p.

        Args:
            index (int | bool, optional): Slice index. Defaults to False, in which case it automatically slices where the wavefunction density os maximal.
            axis (Literal[, optional): Axis perpendicular to the slice plane. Defaults to -1, in which case it uses the stored value of self.slice_axis.
        """
        self.psi2D_s, self.psi2D_p = self.get_2D_wavefunctions(index = index, axis = axis)
        return

    def get_2D_wavefunctions(self, index: int | bool = False, axis: Literal[-1, 0, 1, 2] = -1) -> tuple[np.ndarray, np.ndarray]:
        """Generate 2D numpy arrays representing slices of the wavefunction. This function outputs both the wavefunction itself and its p-wave tip form.

        Args:
            index (int | bool, optional): Slice index. Defaults to False, in which case it automatically slices where the wavefunction density os maximal.
            axis (Literal[, optional): Axis perpendicular to the slice plane. Defaults to -1, in which case it uses the stored value of self.slice_axis.

        Returns:
            tuple[np.ndarray, np.ndarray]: Ordinary s-type wavefunction (cube data) and p-type wavefunction (cube data)
        """
        if not axis in {0, 1, 2}: axis = self.slice_axis # type: ignore
        # Default to the internally stored slice axis
        
        if isinstance(index, bool):
            # If a boolean is provided for index, auto select from the slice that gives the maximum intensity of the wavefunction density
            orb = np.abs(self.psi) ** 2
            match axis:
                case 0: maxima = np.max(orb, axis = (1, 2))
                case 1: maxima = np.max(orb, axis = (0, 2))
                case 2: maxima = np.max(orb, axis = (0, 1))
            index = int(np.argmax(maxima))
        
        match axis:
            case 0: wfn_2D = self.psi[index]
            case 1: wfn_2D = self.psi[:, index]
            case 2: wfn_2D = self.psi[:, :, index]
        
        # Store the sliced wavefunction as the attribute self.psi2D_s. Also generate self.psi2D_p for a p-wave representation
        psi2D_s = wfn_2D
        psi2D_p = sobel(wfn_2D, axis = 1, mode = "wrap") + 1j * sobel(wfn_2D, axis = 0, mode = "wrap")
        return psi2D_s, psi2D_p

    def get_2D_orbital(self, index: int | bool = False, axis: Literal[-1, 0, 1, 2] = -1) -> np.ndarray:
        if not axis in {0, 1, 2}: axis = self.slice_axis # type: ignore
        # Default to the internally stored slice axis
        
        orb = np.abs(self.psi) ** 2
        
        if isinstance(index, bool):
            # Auto select from the slice that gives the maximum intensity of the wavefunction density
            match self.slice_axis:
                case 0: maxima = np.max(orb, axis = (1, 2))
                case 1: maxima = np.max(orb, axis = (0, 2))
                case 2: maxima = np.max(orb, axis = (0, 1))
            index = int(np.argmax(maxima))

        match axis:
            case 0: orb2D = orb[index]
            case 1: orb2D = orb[:, index]
            case 2: orb2D = orb[:, :, index]
        return orb2D

    def generate_thumbnail(self):
        psi2D_s, psi2D_p = self.get_2D_wavefunctions()
        orb = complex_to_rgba(psi2D_s)
        
        fig, ax = plt.subplots(figsize = (3, 3), dpi = 100)
        ax.axis('off')
        fig.patch.set_visible(False)
        ax.imshow(orb)
        
        buffer = BytesIO()
        plt.savefig(buffer, format = "png", bbox_inches = "tight", pad_inches = 0, transparent = True)
        plt.close(fig)
        
        buffer.seek(0)        
        return buffer.read()

    def show(self):
        print(f"{self.info()}")        
        png_bytes = self.generate_thumbnail()
        display.display(display.Image(data = png_bytes, format = "png"))
        return

    @classmethod
    def from_wavecar(cls, wavecar_object: vaspwfc, spin: int = 0, kpoint: int = 0, band: int = 0, target_energy: float | None = None):
        wfc = wavecar_object
        n_kpoints = int(wfc._nkpts)
        n_spins = int(wfc._nspin)
        n_bands = int(wfc._nbands)
        latvecs = wfc._Acell
        
        if not isinstance(spin, int) or not spin in {0, 1}: raise Exception(f"Invalid spin index {spin}. Only 0 and 1 are recognized.")
        if spin == 1 and n_spins < 2: raise Exception(f"Invalid spin index {spin} for a wavecar object that has no spin polarization ({n_spins = }).")
        if not isinstance(kpoint, int) or kpoint > n_kpoints - 1: raise Exception(f"Invalid k-point index {kpoint} for a wavecar object containing {n_kpoints} k-points.")
        if not isinstance(target_energy, float):
            if not isinstance(band, int) or band > n_bands - 1: raise Exception(f"Invalid band index {band} for a wavecar object containing {n_bands} bands.")
       
        try:
            kvecs_reciprocal = np.round(wfc._kvecs, 7)
            kvec_reciprocal = kvecs_reciprocal[kpoint]
            kvecs = wfc._Bcell
            kvecs_cartesian = np.dot(kvecs, kvecs_reciprocal.T).T
            kvec_cartesian = kvecs_cartesian[kpoint]
        
            all_energies = wfc._bands
            assert isinstance(all_energies, np.ndarray)
            energies_at_spin_and_kpoint = all_energies[spin, kpoint]
            
            if isinstance(target_energy, float): # If a target energy is provided, find the eigenstate closest to that energy                
                energy_differences = np.abs(energies_at_spin_and_kpoint - target_energy)
                band = int(np.argmin(energy_differences))
            
            eigenenergy = float(energies_at_spin_and_kpoint[band])
        except Exception as e:
            raise Exception(f"Unable to extract the eigenenergy from the wavecar object: {e}")
        
        try:
            psi = wfc.wfc_r(ispin = spin + 1, ikpt = kpoint + 1, iband = band + 1)
            assert isinstance(psi, np.ndarray)
        except Exception as e:
            raise Exception(f"Unable to obtain the wavefunction from the wavecar object: {e}")
        
        return cls(eigenenergy, spin, band, psi, latvecs, kvec_reciprocal, kvec_cartesian, kpoint)

