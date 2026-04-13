import os
import matplotlib.pyplot as plt
from matplotlib import colors as cols
import numpy as np
from . import vaspwfc
from pymatgen.core import Lattice, Structure, Molecule
import nglview as nv



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

    def get_wavecar(self, path: str, lgamma: bool = True) -> vaspwfc:
        try:
            wfc = vaspwfc(path, lgamma = lgamma)
            return wfc
        except Exception as e:
            print("Error loading the wavecar")
            return False

    def get_structure(self, path: str) -> Structure:
        try:
            structure = Structure.from_file(path)
            return structure
        except Exception as e:
            print("Error loading the wavecar")
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
        all_bands = wavecar_object._bands
        all_band_occs = wavecar_object._occs
        
        [bands_up, bands_down] = [all_bands[i] for i in range(2)]
        [occs_up, occs_down] = [all_band_occs[i] for i in range(2)]
        
        [bands_up_gamma, bands_down_gamma] = [bands_up[0], bands_down[0]] # Assuming k-point 0 is the Gamma point
        [occs_up_gamma, occs_down_gamma] = [occs_up[0], occs_down[0]]
        
        LUMO_up_index = int(np.where(occs_up_gamma < .5)[0][0])
        HOMO_up_index = LUMO_up_index - 1
        LUMO_down_index = int(np.where(occs_down_gamma < .5)[0][0])
        HOMO_down_index = LUMO_down_index - 1
        
        HOMO_up_energy = float(bands_up_gamma[HOMO_up_index])
        HOMO_down_energy = float(bands_down_gamma[HOMO_down_index])
        LUMO_up_energy = float(bands_up_gamma[LUMO_up_index])
        LUMO_down_energy = float(bands_down_gamma[LUMO_down_index])
        
        return {"HOMO_up_index": HOMO_up_index, "HOMO_down_index": HOMO_down_index, "LUMO_up_index": LUMO_up_index, "LUMO_down_index": LUMO_down_index,
                "HOMO_up_energy": HOMO_up_energy, "HOMO_down_energy": HOMO_down_energy, "LUMO_up_energy": LUMO_up_energy, "LUMO_down_energy": LUMO_down_energy}

    def spin_and_occupation_resolved_DOS(self, wavecar_object: vaspwfc, *args, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weights = kwargs.pop("weights", None)
       
        all_bands = wavecar_object._bands
        all_band_occs = wavecar_object._occs
        
        [bands_up, bands_down] = [all_bands[i] for i in range(2)]
        [occs_up, occs_down] = [all_band_occs[i] for i in range(2)]
        
        [bands_up_gamma, bands_down_gamma] = [bands_up[0], bands_down[0]] # Assuming k-point 0 is the Gamma point
        [occs_up_gamma, occs_down_gamma] = [occs_up[0], occs_down[0]]

        LDOS_up_occ = self.DOS_from_energies(bands_up_gamma, weights = occs_up_gamma, *args, **kwargs)
        LDOS_up_unocc = self.DOS_from_energies(bands_up_gamma, weights = 1 - occs_up_gamma, *args, **kwargs)
        LDOS_down_occ = self.DOS_from_energies(bands_down_gamma, weights = occs_down_gamma, *args, **kwargs)
        LDOS_down_unocc = self.DOS_from_energies(bands_down_gamma, weights = 1 - occs_down_gamma, *args, **kwargs)
        
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
        ax.fill_betweenx(LDOS_up_occ[0], LDOS_up_occ[1], color = col_up_occ)
        ax.fill_betweenx(LDOS_up_unocc[0], LDOS_up_unocc[1], color = col_up_unocc)
        ax.fill_betweenx(LDOS_down_occ[0], -LDOS_down_occ[1], color = col_down_occ)
        ax.fill_betweenx(LDOS_down_unocc[0], -LDOS_down_unocc[1], color = col_down_unocc)
        
        return fig

