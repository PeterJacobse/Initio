import os, re, time, threading, logging, yaml, shutil
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors as cols
import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter, zoom, sobel
from PIL import Image
import pymatgen.core.structure as pmg_struct
from pymatgen.core import periodic_table
from pymatgen.io import vasp, ase
from ase.io import read
import nglview as nv
from ase.neighborlist import NeighborList
from ase.data.colors import jmol_colors
from unfold import make_kpath, find_K_from_k, removeDuplicateKpoints, unfold
from . import vaspwfc
from collections.abc import Iterable



def find_folder(target: str = "", base_folder = "C:\\DFT") -> str:
    target_folder = None
    sub_folders = [os.path.join(base_folder, folder) for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    for sub_folder in sub_folders:
        if os.path.isdir(os.path.join(sub_folder, target)): target_folder = os.path.join(sub_folder, target)
    if not target_folder:
        for sub_folder in sub_folders:
            subsub_folders = [os.path.join(sub_folder, folder) for folder in os.listdir(sub_folder) if os.path.isdir(os.path.join(sub_folder, folder))]
            for subsub_folder in subsub_folders:
                if os.path.isdir(os.path.join(subsub_folder, target)): target_folder = os.path.join(subsub_folder, target)

    target_folder
    if not target_folder: raise Exception("Folder not found")
    return target_folder

def autocrop_image(file_path: str = "") -> None:
    if not os.path.isfile(file_path):
        logging.info("Invalid file path passed to autocrop_image")
        return
    
    try:
        img = Image.open(file_path).convert("RGBA")
        alpha_mapped = img.convert("RGBa")
        bbox = alpha_mapped.getbbox()

        if bbox:
            cropped_img = img.crop(bbox)
            cropped_img.save(file_path)
            logging.info(f"Cropped image from size {img.size} to size {cropped_img.size}.")
        else:
            logging.info("Unable to crop the image")
    except:
        logging.info("Unable to crop the image")
    return

def save_image(view: nv.NGLWidget, file_path: str = ""):
    def save_image_when_ready(img_widget, file_path: str = ""):
        img_bytes = None
        
        while not img_widget.value: time.sleep(.1)
        img_bytes = img_widget.value.tobytes()
        
        if not img_bytes:
            logging.info("Error: Could not save image.")
            return

        with open(file_path, "wb") as f:
            f.write(img_bytes)
        logging.info(f"Image saved to {file_path}")
        
        autocrop_image(file_path)
        return
    
    img_widget = view.render_image(transparent = True)
    monitor_thread = threading.Thread(target = save_image_when_ready, args = (img_widget, file_path))
    monitor_thread.daemon = True
    monitor_thread.start()
    return

def gaussian_smearing_org(x, x0, sigma = 0.05):
    '''
    Gaussian smearing of a Delta function.
    '''
    return 1. / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def string2index(string):
    if ':' not in string:
        raise ValueError("Invalid slice string!")
    i = []
    for s in string.split(':'):
        if s == '':
            i.append(None)
        else:
            i.append(int(s))
    i += (3 - len(i)) * [None]
    return slice(*i)

def gradient_fill(x, y, fill_color = None, ax = None, direction = 1, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """

    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    # print fill_color
    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:, :, :3] = rgb
    if direction == 1:
        z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]
    else:
        z[:, :, -1] = np.linspace(alpha, 0, 100)[:, None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    if direction == 1:
        xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    else:
        xy = np.vstack([[xmin, ymax], xy, [xmax, ymax], [xmin, ymax]])
    clip_path = Polygon(xy, lw=0.0, facecolor='none',
                        edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)

    return line, im



class UnfoldProcar(object):
    '''
    A class for dealing with VASP PROCAR file.
    '''

    def __init__(self, inf = "PROCAR", lsoc = False):
        '''
        Initialization
        '''

        self._fname = inf
        # the directory containing the input file
        self._dname = os.path.dirname(inf)
        if self._dname == '':
            self._dname = '.'

        self._lsoc = lsoc

        try:
            self._procar = open(self._fname, 'r')
        except:
            raise IOError('Failed to open %s' % self._fname)

        self.readProcar()

        # parameters usefull for dos generation
        self._sigma = 0.05
        self._nedos = 3000
        # Total DOS for each KS energy, with shape (NSPIN, NKPTS, NBANDS, NEDOS)
        self._tdos = None
        # Total DOS with shape (NSPIN, NEDOS)
        self._totalDOS = None

        self._spd_index = {
            's': 0,
            'py': 1, 'pz': 2, 'px': 3,
            'dxy': 4, 'dyz': 5, 'dz2': 6, 'dxz': 7, 'dx2': 8
        }

        # the basis vectors of the cell
        self._cell = None
        self._kpath = None

    def readProcar(self):
        '''
        Extract the info from PROCAR.
        '''

        inp = [line for line in self._procar if line.strip()]

        # when the band number is too large, there will be no space between ";" and
        # the actual band number. A bug found by Homlee Guo.
        # Here, #kpts, #bands and #ions are all integers
        self._nkpts, self._nbands, self._nions = [
            int(xx) for xx in re.sub('[^0-9]', ' ', inp[1]).split()]

        # band projectron on each atoms or s/p/d orbitals
        self._aproj = np.asarray([line.split()[1:-1] for line in inp
                                  if not re.search('[a-zA-Z]', line)],
                                 dtype=float)
        # k-points weights of each k-points
        # self._kptw = np.asarray([line.split()[-1]
        # In case that there are no blank spaces before the '-' sign of the
        # k-points coordinates
        self._kptw = np.asarray([re.sub(r'(\d)-', r'\1 -', line).split()[-1]
                                 for line in inp if 'weight' in line], dtype=float)
        # k-points vectors of each k-points
        # self._kptv = np.asarray([line.split()[-6:-3]
        # In case that there are no blank spaces before the '-' sign of the
        # k-points coordinates
        self._kptv = np.asarray([re.sub(r'(\d)-', r'\1 -', line).split()[-6:-3]
                                 for line in inp if 'weight' in line], dtype=float)
        # in case of spin poliarized calculation
        self._kptv = self._kptv[:self._nkpts]
        # band energies
        self._eband = np.asarray([line.split()[-4] for line in inp
                                  if 'occ.' in line], dtype=float)

        self._nlmax = self._aproj.shape[-1]
        self._nspin = self._aproj.shape[0] // (
            self._nkpts * self._nbands * self._nions)
        self._nspin //= 4 if self._lsoc else 1

        if self._lsoc:
            self._aproj.resize(self._nspin, self._nkpts,
                               self._nbands, 4, self._nions, self._nlmax)
            self._Mxyz = self._aproj[:, :, :, 1:, :, :]
            self._aproj = self._aproj[:, :, :, 0, :, :]
        else:
            self._aproj.resize(self._nspin, self._nkpts,
                               self._nbands, self._nions, self._nlmax)

        self._kptw.shape = (self._nspin, self._nkpts)
        self._kptw_org = self._kptw.copy()
        self._eband.shape = (self._nspin, self._nkpts, self._nbands)

        # close the PROCAR
        self._procar.close()

    def get_magnetization(self):
        '''
        In an noncollinear calculation, get the magnetization in x/y/z
        direction.
        '''

        assert self._lsoc

        return self._Mxyz.copy()

    def get_nkpts(self):
        '''
        get number of kpoints.
        '''
        return self._nkpts

    def get_nspin(self):
        '''
        get number of spin
        '''
        return self._nspin

    def get_nbands(self):
        '''
        get number of bands
        '''
        return self._nbands

    def get_band_energies(self):
        '''
        Return the band energies
        '''
        return self._eband.copy()

    def get_kpath(self, cell = None, nkseg = None):
        '''
        Construct k-point path, find out the k-path boundary if possible.
        '''

        if self._kpath is None:
            if self._cell is None:
                if cell is None:
                    try:
                        self._cell = read(
                            self._dname + '/POSCAR', format='vasp').cell.copy()
                    except:
                        raise ValueError(
                            'Error in reading cell info from POSCAR!')
                else:
                    self._cell = np.array(cell, dtype=float)
                    assert self._cell.shape == (3, 3)

            if nkseg is None:
                if os.path.isfile(self._dname + "/KPOINTS"):
                    kfile = open(self._dname + "/KPOINTS").readlines()
                    if kfile[2][0].upper() == 'L':
                        nkseg = int(kfile[1].split()[0])
                    else:
                        raise ValueError(
                            'Error reading number of k-points from KPOINTS')

            assert isinstance(nkseg, int) and nkseg > 0

            nsec = self._nkpts // nkseg
            icell = np.linalg.inv(self._cell).T

            # vkpts_d = np.diff(self._kptv, axis=0)
            # self._kpath     = np.zeros(self._nkpts, dtype=float)
            # self._kpath[1:] = np.cumsum(np.linalg.norm(np.dot(vkpts_d, icell), axis=1))

            v = self._kptv.copy()
            for ii in range(nsec):
                ki = ii * nkseg
                kj = (ii + 1) * nkseg
                v[ki:kj, :] -= v[ki]

            self._kpath = np.linalg.norm(np.dot(v, icell), axis=1)
            for ii in range(1, nsec):
                ki = ii * nkseg
                kj = (ii + 1) * nkseg
                self._kpath[ki:kj] += self._kpath[ki - 1]

            self._kbound = np.concatenate(
                (self._kpath[0::nkseg], [self._kpath[-1], ]))

        return self._kpath, self._kbound

    def isSoc(self):
        return True if self._lsoc else False

    def get_sigma(self):
        '''
        return dos brodening parameter
        '''
        return self._sigma

    def set_sigma(self, sigma):
        '''
        set dos brodening parameter
        '''
        self._sigma = sigma

        # re-generate the DOS with the new SIGMA
        if self._tdos is not None:
            if not np.isclose(sigma, self._sigma):
                self.init_dos()

    def get_nedos(self):
        return self._nedos

    def set_nedos(self, nedos):
        '''
        set number of point in smooth DOS
        '''
        assert isinstance(nedos, int), 'NEDOS shoule be int!'
        self._nedos = nedos

        # re-generate the DOS with the new NEDOS
        if self._tdos is not None:
            if self._nedos != nedos:
                self.init_dos()

    def get_kpts_weight(self):
        '''
        return the k-points weights
        '''
        return self._kptw.copy()

    def set_kpts_weight(self, kptw):
        '''
        set the k-points weights
        '''
        kptw = np.array(kptw)
        assert kptw.shape == self._kptw.shape
        self._kptw = kptw

        # re-generate the DOS with the new kptw
        if self._tdos is not None:
            self.init_dos()

    def restore_kpts_weight(self, kptw):
        '''
        set the k-points weights
        '''
        self._kptw = self._kptw_org.copy()

        # re-generate the DOS with the new kptw
        if self._tdos is not None:
            self.init_dos()

    def init_dos(self):
        '''
        dos initialization
        '''

        # print 'calculating dos'
        emin = self._eband.min()
        emax = self._eband.max()
        eran = emax - emin
        emin = emin - eran * 0.05
        emax = emax + eran * 0.05

        self._xen = np.linspace(emin, emax, self._nedos)
        self._tdos = np.empty(
            (self._nspin, self._nkpts, self._nbands, self._nedos))

        for ispin in range(self._nspin):
            sign = 1 if ispin == 0 else -1
            for ikpt in range(self._nkpts):
                for iband in range(self._nbands):
                    x0 = self._eband[ispin, ikpt, iband]
                    self._tdos[ispin, ikpt, iband] = sign * self._kptw[ispin, ikpt] \
                        * gaussian_smearing_org(self._xen, x0, self._sigma)\

    def translate_selection(self, atoms = ':', kpts = ':', spd = ':'):
        '''
        '''
        # string is Iterable too
        assert (isinstance(atoms, int)
                or isinstance(atoms, Iterable)
                or isinstance(atoms, str))
        assert (isinstance(kpts, int)
                or isinstance(kpts, Iterable)
                or isinstance(kpts, str))
        assert (isinstance(spd, int)
                or isinstance(spd, Iterable)
                or isinstance(kpts, str))

        if isinstance(atoms, int):
            atoms = [atoms]
        if isinstance(kpts, int):
            kpts = [kpts]
        if isinstance(spd, int):
            spd = [spd]

        if isinstance(atoms, str):
            atoms = string2index(atoms)
        if isinstance(kpts, str):
            kpts = string2index(kpts)
        if isinstance(spd, str):
            spd = string2index(spd)

        # remove duplicate selections
        if isinstance(atoms, Iterable):
            atoms = list(set(atoms))
        if isinstance(kpts, Iterable):
            kpts = list(set(kpts))
        if isinstance(spd, Iterable):
            spd = [ii if isinstance(ii, int) else self._spd_index[ii]
                   for ii in spd]
            spd = list(set(spd))

        return atoms, kpts, spd

    def get_proj(self):
        '''
        get the partial weight.
        '''
        return self._aproj.copy()

    def get_total_dos(self):
        '''
        The total DOS
        '''
        if self._tdos is None:
            self.init_dos()

        if self._totalDOS is None:
            self._totalDOS = np.sum(self._tdos, axis=(1, 2))

        return self._xen, self._totalDOS

    def get_pw(self, atoms = ':', kpts = ':', spd = ':'):
        '''
        Get site/k-points/spd-orbital projected weight for each KS orbital.

        atoms : selected atoms index.
                Valid values:
                    ":"       -> for all atoms
                    "0::2"    -> for even index atoms
                    [0, 1, 2] -> atom indices specified by list
                    0         -> atom indices specified by integer

        kpts  : selected k-points index
                Valid values:
                    ":"       -> for all k-points
                    "0::2"    -> for even index k-points
                    [0, 1, 2] -> k-points indices specified by list
                    0         -> k-points indices specified by integer

        spd   : selected s/p/d-orbitals, the s/p/d-orbital and the corresponding
                index are:
                    's' : 0,
                    'py' : 1, 'pz' : 2, 'px' : 3,
                    'dxy' : 4, 'dyz' : 5, 'dz2' : 6, 'dxz' : 7, 'dx2' : 8

                Valid values:
                    ":"         -> for all s/p/d-orbitals
                    "0::2"      -> for even index
                    [0, 1, 2]   -> s/p/d-orbitals specified by list of integer
                    ['s', 'py'] -> s/p/d-orbitals specified by list of names
                    0           -> s/p/d-orbitals indices specified by integer
        '''

        atoms, kpts, spd = self.translate_selection(atoms, kpts, spd)

        # problem with mixed advanced indexing and basic indexing, see scipy
        # documents for reference
        # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#combining-advanced-and-basic-indexing
        #
        # a=np.zeros((2,3,4)); b=np.ones((3,4)); I=np.array([0,1])
        # b[:,I].shape = (3, 2)
        # a[0,:,I].shape = (2, 3)

        # Consider indexing a 3D array arr with shape (X, Y, Z):
        #
        # arr[:, [0, 1], 0] has shape (X, 2).
        # arr[[0, 1], 0, :] has shape (2, Z).
        # arr[0, :, [0, 1]] has shape (2, Y), not (Y, 2)

        pw = []
        for ispin in range(self._nspin):
            p0 = self._aproj[ispin, kpts]
            # sum over the s/p/d projection
            p0 = np.sum(p0[..., spd],   axis=-1)
            # sum over the site projection
            p0 = np.sum(p0[..., atoms], axis=-1)

            pw.append(p0)

        return np.array(pw, dtype=float)

    def get_pdos(self, atoms = ':', kpts = ':', spd = ':'):
        '''
        Get site/k-points/spd-orbital projected partial density of states (PDOS)
        '''

        if self._tdos is None:
            self.init_dos()

        pdos = []
        proj = self.get_pw(atoms, kpts, spd)

        atoms, kpts, spd = self.translate_selection(atoms, kpts, spd)

        if len(np.arange(self._nkpts)[kpts]) == self._nkpts:
            if np.all(
                np.sort(np.arange(self._nkpts)[kpts]) == np.arange(self._nkpts)
            ):
                used_all_kpts = True
            else:
                used_all_kpts = False
        else:
            used_all_kpts = False

        for ispin in range(self._nspin):
            pw = proj[ispin]

            if used_all_kpts:
                td = self._tdos[ispin, kpts]
            else:
                # if not all the k-points are used, then probably we should get
                # rid of the k-point weights
                td = self._tdos[ispin, kpts, ...] / \
                    self._kptw[ispin, kpts, np.newaxis, np.newaxis]

            pdos.append(np.sum(pw[..., np.newaxis] * td, axis=(0, 1)))

            # pwht = np.sum(self._aproj[ispin][kpts,:,atoms,spd], axis=(-1, -2))
            # pdos.append(np.sum(pwht[..., np.newaxis] * self._tdos[ispin][kpts,...], axis=(0, 1)))

        # only return one dos if not spin-polarized
        # p = pdos[0] if self._nspin == 1 else pdos
        pdos = np.array(pdos, dtype=float)

        return self._xen, pdos

    def get_pband(self, atoms = ':', kpts = ':', spd = ':', cell = None, nkseg = None):
        '''
        Construct the band structure from PROCAR. In addition, the
        site/k-points/spd-orbital projection of each KS orbital will be
        returned.
        '''

        k, b = self.get_kpath(cell, nkseg)
        e = self.get_band_energies()
        w = self.get_pw(atoms, kpts, spd)

        return k, b, e, w



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



    # IO
    def read_vasp_file(self, path: str) -> vaspwfc | Initio.Structure | bool:
        base_name = os.path.basename(path)
        
        match base_name:
            case "WAVECAR": return self.get_wavecar(path)
            case "POSCAR" | "CONTCAR": return self.get_structure(path)
            case "PROCAR": return self.get_procar(path)
            case "INCAR": return self.get_incar(path)
            case "POTCAR": return self.get_potcar(path)
            case "KPOINTS": return self.get_kpoints(path)
            case "EIGENVAL": return self.get_eigenval(path)
            case "OUTCAR": return self.get_outcar(path)
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
            print(f"Could not open POSCAR / CONTCAR file: {e}")
            return False

    def get_incar(self, path: str) -> vasp.inputs.Incar:
        try:
            incar = vasp.Incar.from_file(path)
            return incar
        except Exception as e:
            print(f"Could not opten INCAR file: {e}")
            return False

    def get_potcar(self, path: str) -> vasp.inputs.Potcar:
        try:
            potcar = vasp.Potcar.from_file(path)
            return potcar
        except Exception as e:
            print(f"Could not open POTCAR file: {e}")
            return False

    def get_procar(self, path: str) -> UnfoldProcar:
        try:
            procar = UnfoldProcar(path)
            return procar
        except Exception as e:
            print(f"Could not open PROCAR file: {e}")
            return False

    def get_kpoints(self, path: str) -> vasp.inputs.Kpoints:
        try:
            kpoints = vasp.Kpoints.from_file(path)
            return kpoints
        except Exception as e:
            print(f"Could not open KPOINTS file: {e}")
            return False

    def get_eigenval(self, path: str) -> vasp.outputs.Eigenval:
        try:
            eigenval = vasp.outputs.Eigenval(path)
            return eigenval
        except Exception as e:
            print(f"Could not open EIGENVAL file: {e}")
            return False

    def get_outcar(self, path: str) -> vasp.outputs.Outcar:
        try:
            outcar = vasp.outputs.Outcar(path)
            return outcar
        except Exception as e:
            print(f"Could not open OUTCAR file: {e}")
            return False

    def show_incar(self, incar: vasp.inputs.Incar) -> None:
        try:
            print(incar.get_str(pretty = True))
        except Exception as e:
            print(f"Error: {e}")
        return



    # Structure manipulation
    def combine_structures(self, structure1, structure2, translate1: list = [0, 0, 0], translate2: list = [0, 0, 0]) -> Initio.Molecule:
        if not isinstance(structure1, Initio.Molecule | pmg_struct.Molecule | Initio.Structure | pmg_struct.Structure):
            print(f"Unsupported type for structure 1: {type(structure1)}")
            return
        if not isinstance(structure2, Initio.Molecule | pmg_struct.Molecule | Initio.Structure | pmg_struct.Structure):
            print(f"Unsupported type for structure 2: {type(structure2)}")
            return
        
        struc1_copy = structure1.copy()
        struc2_copy = structure2.copy()
        if np.sum(np.abs(translate1)) > .0001: struc1_copy.translate_sites(range(len(struc1_copy)), translate1)
        if np.sum(np.abs(translate2)) > .0001: struc2_copy.translate_sites(range(len(struc2_copy)), translate2)
        
        all_species = list(struc1_copy.species) + list(struc2_copy.species)
        all_coords = list(struc1_copy.cart_coords) + list(struc2_copy.cart_coords)
        
        molecule = Initio.Molecule(all_species, all_coords)
        return molecule

    def make_polyhedron(self, structure, sides: int = 5, size: int | float = 10., center_xy: list = [0, 0], start_angle_deg: int | float = 0.) -> Initio.Molecule:
        if not isinstance(structure, Initio.Molecule | pmg_struct.Molecule | Initio.Structure | pmg_struct.Structure):
            print(f"Unsupported type for structure: {type(structure)}")
            return
        
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / sides)
        angles += np.deg2rad(start_angle_deg)
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        new_species = []
        new_coords = []
        for specie, coord in zip(list(structure.species), list(structure.cart_coords)):
            atom_xy = coord[:2] - center_xy
            projections = np.dot(normals, atom_xy)
            
            if np.all(projections <= size):
                new_species.append(specie)
                new_coords.append(coord)
        
        molecule = self.Molecule(species = new_species, coords = new_coords)
        return molecule



    # Energy data manipulation
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
        
        HOMO_up_index = int(np.where(occs_up < .5)[0][0])
        LUMO_up_index = HOMO_up_index + 1
        HOMO_down_index = int(np.where(occs_down < .5)[0][0])
        LUMO_down_index = HOMO_down_index + 1        
        
        HOMO_up_energy = float(bands_up[HOMO_up_index] - 1)
        HOMO_down_energy = float(bands_down[HOMO_down_index] - 1)
        LUMO_up_energy = float(bands_up[LUMO_up_index] - 1)
        LUMO_down_energy = float(bands_down[LUMO_down_index] - 1)
        
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

    def get_band_indices(self, calculation_path: str, energy_range: list | np.ndarray) -> tuple[float, int, int]:
        try:
            if not isinstance(calculation_path, str) or not os.path.isdir(calculation_path):
                print("No valid calculation path given")
                raise Exception()
            
            eigenval = self.read_vasp_file(os.path.join(calculation_path, "EIGENVAL"))
            outcar = vasp.outputs.Outcar(os.path.join(calculation_path, "OUTCAR"))
            
            Fermi_level = outcar.efermi
            first_spin_channel = list(eigenval.eigenvalues.keys())[0]
            gamma_bands = eigenval.eigenvalues[first_spin_channel][0]

            # Enumerate to track VASP band indices (1-indexed for standard VASP reference)
            matching_indices = []
            for idx, band in enumerate(gamma_bands, start=1):
                energy = band[0]
                if np.min(energy_range) <= energy <= np.max(energy_range):
                    matching_indices.append(idx)

            if not matching_indices:
                return (Fermi_level, 0, 0)

            min_band = min(matching_indices)
            max_band = max(matching_indices)
            
            return (Fermi_level, min_band, max_band)
        
        except Exception as e:
            print("Error trying to retrieve band data from the EIGENVAL and OUTCAR")
            return (0, 0, 0)

    def clean_kpath(self, kpath: list | np.ndarray, crystal_type: str = "hexagonal") -> np.ndarray:
        if not isinstance(kpath, list | np.ndarray):
            raise Exception(f"Invalid k-path provided to clean-kpath: {kpath}")
        
        cleaned_kpath = []
        for kpoint in kpath:
            match kpoint:
                case str() if kpoint.lower() == "gamma": cleaned_kpath.append([0, 0, 0])
                case str() if kpoint.lower() == "m": cleaned_kpath.append([.5, 0, 0])
                case str() if kpoint.lower() == "k":
                    if crystal_type == "hexagonal": cleaned_kpath.append([1/3, 1/3, 0])
                case str() if kpoint.lower() == "x":
                    if crystal_type == "hexagonal": cleaned_kpath.append([0, .5, .5])
                    elif crystal_type == "orthorhombic": cleaned_kpath.append([.5, 0, 0])
                case str() if kpoint.lower() == "a": cleaned_kpath.append([0, 0, .5])
                case str() if kpoint.lower() == "l":
                    if crystal_type == "hexagonal": cleaned_kpath.append([.5, 0, .5])            
                case str() if kpoint.lower() == "h":
                    if crystal_type == "hexagonal": cleaned_kpath.append([1/3, 1/3, .5])
                case str() if kpoint.lower() == "y":
                    if crystal_type == "orthorhombic": cleaned_kpath.append([0, .5, 0])
                case str() if kpoint.lower() == "z":
                    if crystal_type == "orthorhombic": cleaned_kpath.append([0, 0, .5])
                case list():
                    if len(kpoint) == 3 and isinstance(kpoint[0], float | int): cleaned_kpath.append(kpoint)
                case _:
                    print(f"Ignoring unrecognized kpoint in kpath: {kpoint}")
        
        return np.array(cleaned_kpath, dtype = float)



    # Visualization
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



    # Unfolding
    def find_supercell_matrix(self, primitive_structure: Initio.Structure, supercell_structure: Initio.Structure) -> tuple[np.ndarray, np.ndarray]:
        try:
            prim_vecs = primitive_structure.lattice.matrix
            supercell_vecs = supercell_structure.lattice.matrix
            M_raw = np.dot(supercell_vecs, np.linalg.inv(prim_vecs))
            return (np.astype(np.round(M_raw), int), M_raw)
        except:
            return False

    def generate_k_mapping(self, transformation_matrix: np.ndarray = np.eye(3), kpath: list | np.ndarray = [], crystal_type: str = "hexagonal", nseg: int = 12) -> tuple[np.ndarray, np.ndarray]:
        k_path_fractional = initio.clean_kpath(kpath = kpath, crystal_type = crystal_type)

        kpts_prim = make_kpath(k_path_fractional, nseg = nseg) # Structure of the KPOINTS as required by the VASPbandunfolding library
        K_in_sup = []
        for k in kpts_prim:
            K, G = find_K_from_k(k, transformation_matrix)
            K_in_sup.append(K)
        reduced_K, k_map = removeDuplicateKpoints(K_in_sup, return_map = True)
        return (reduced_K, k_map)

    def test_k_mappings(self, transformation_matrix: np.ndarray, kpath: list | np.ndarray = [], crystal_type: str = "", npoints_max: int = 40):
        lengths = []
        npoints = []
        for n in range(npoints_max):
            (reduced_K, k_map) = initio.generate_k_mapping(transformation_matrix, kpath = kpath, crystal_type = crystal_type, nseg = n)
            lengths.append(len(k_map))
            npoints.append(n)
        return (npoints, lengths)

    def generate_unfolding_calculation(self, supercell_folder: str = None, primitive_folder: str = None, kpath: list | np.ndarray = ["gamma", "k", "m", "gamma"], crystal_type: str = "hexagonal", nseg: int = 12, suppress: bool = False) -> None:
        try:
            if not isinstance(supercell_folder, str) or not os.path.isdir(supercell_folder):
                raise Exception("Invalid supercell calculation folder")
            if not isinstance(primitive_folder, str) or not os.path.isdir(primitive_folder):
                raise Exception("Invalid primitive calculation folder")
            
            
            
            # Create the 'scf' and 'unfolding' folders and copy the structures
            contcar_file = os.path.join(supercell_folder, "CONTCAR")
            if not os.path.isfile(contcar_file):
                raise Exception("No valid CONTCAR file found of the supercell")
            
            supercell_struc: Initio.Structure = self.read_vasp_file(contcar_file)
            scf_folder = os.path.join(supercell_folder, "scf")
            unfolding_folder = os.path.join(supercell_folder, "unfolding")
            
            if not suppress:
                if not os.path.isdir(scf_folder): os.mkdir(scf_folder) # Create the SCF folder
                supercell_struc.to_file(os.path.join(scf_folder, "POSCAR"))
                
                if not os.path.isdir(unfolding_folder): os.mkdir(unfolding_folder) # Create the unfolding folder            
                supercell_struc.to_file(os.path.join(unfolding_folder, "POSCAR"))

            
            
            # Retrieve the supercell transformation matrix from the supercell structure and the primitive cell structure
            contcar_file = os.path.join(primitive_folder, "CONTCAR")
            if not os.path.isfile(contcar_file):
                raise Exception("No valid CONTCAR file found of the primitive cell")
            
            prim_struc: Initio.Structure = self.read_vasp_file(contcar_file)
            (transformation_matrix, transformation_matrix_raw) = self.find_supercell_matrix(prim_struc, supercell_struc)
            print("\nI found the following transformation matrix from primitive to supercell:\n")
            print(f"{transformation_matrix_raw}")
            
            
            
            # Get the INCAR from the supercell calculation            
            supercell_incar: vasp.Incar = self.read_vasp_file(os.path.join(supercell_folder, "INCAR"))            
            print("\nI found the following INCAR parameters for the supercell calculation:\n")
            self.show_incar(supercell_incar)
            
            # Adapt the INCAR for the single-point SCF calculation and subsequent unfolding calculation
            supercell_incar["LCHARG"] = True
            supercell_incar["LORBIT"] = False
            supercell_incar["LWAVE"] = False # Do not generate a WAVECAR at this time. Instead, use the Eigenvalues to parse which bands to use later and generate the WAVECAR only at the unfolding stage.
            supercell_incar["NSW"] = 0
            supercell_incar["IBRION"] = -1
            if not suppress: supercell_incar.write_file(os.path.join(scf_folder, "INCAR"))

            supercell_incar["LCHARG"] = False
            supercell_incar["ICHARG"] = 11
            supercell_incar["LORBIT"] = 11
            supercell_incar["LWAVE"] = True            
            if not suppress:
                supercell_incar.write_file(os.path.join(unfolding_folder, "INCAR"))
                print("\nSingle-point SCF calculation INCAR file written to 'scf' directory, and NSCF calculation INCAR written to 'unfolding' directory.\n")
            
            
            
            # Copy the POTCARs
            potcar_file = os.path.join(supercell_folder, "POTCAR")
            if not os.path.isfile(potcar_file):
                raise Exception("Missing POTCAR")
            supercell_potcar: vasp.inputs.Potcar = self.read_vasp_file(potcar_file)
            if not suppress:
                supercell_potcar.write_file(os.path.join(scf_folder, "POTCAR"))
                supercell_potcar.write_file(os.path.join(unfolding_folder, "POTCAR"))


            
            # Create the KPOINTS files
            kpoints_file = os.path.join(supercell_folder, "KPOINTS")
            if os.path.isfile(kpoints_file):
                supercell_kpoints: vasp.Kpoints = self.read_vasp_file(kpoints_file)
            else: # No KPOINTS found: fall back to generating a gamma-point calculation
                print("No KPOINTS found in supercell calculation.\nFalling back to generating a gamma-point calculation file")
                supercell_kpoints = vasp.inputs.Kpoints.gamma_automatic()
            if not suppress: supercell_kpoints.write_file(os.path.join(scf_folder, "KPOINTS"))
            
            kpath = self.clean_kpath(kpath, crystal_type = crystal_type)            
            (reduced_K, k_map) = self.generate_k_mapping(transformation_matrix, kpath = kpath, crystal_type = crystal_type, nseg = nseg)            
            kpts_prim = make_kpath(kpath, nseg = nseg) # Structure of the KPOINTS as required by the VASPbandunfolding library
            K_in_sup = []
            for k in kpts_prim:
                K, G = find_K_from_k(k, transformation_matrix)
                K_in_sup.append(K)
            reduced_K, k_map = removeDuplicateKpoints(K_in_sup, return_map = True)
            weights = [1] * len(reduced_K)
            
            kpoints_obj = vasp.Kpoints(comment = "Unfolding supercell k-points generated via VaspBandUnfolding and Initio", num_kpts = len(reduced_K),
                                       style = vasp.Kpoints.supported_modes.Reciprocal, kpts = reduced_K, kpts_weights = weights)
            if not suppress: kpoints_obj.write_file(os.path.join(unfolding_folder, "KPOINTS"))
            
            if not suppress:
                with open(os.path.join(unfolding_folder, "unfolding_parameters.yml"), "w") as f:
                    yaml.safe_dump({"transformation_matrix": transformation_matrix.tolist(), "k_map": k_map.tolist()}, f)
            
            if not suppress:
                print("\nI found the structure and POTCAR of the supercell calculation and copied them to the new folders 'scf' and 'unfolding'.")
                print("Please submit the job in the 'scf' folder to VASP first.")
                print("When completed, run Initio.scf_to_unfolding to finish setting up the 'unfolding' folder for the band unfolding calculation.")
            
        except Exception as e:
            print(f"Error encountered while generating an unfolding claculation: {e}")
        
        return

    def scf_to_unfolding(self, supercell_folder: str = None) -> None:
        if not isinstance(supercell_folder, str) or not os.path.isdir(supercell_folder): raise IOError("Invalid supercell calculation folder")
        
        scf_folder = os.path.join(supercell_folder, "scf")
        unfolding_folder = os.path.join(supercell_folder, "unfolding")
        if not os.path.isdir(scf_folder): raise IOError("'scf' folder not found")
        if not os.path.isdir(unfolding_folder): raise IOError("'unfolding' folder not found")
        
        try:
            (E_Fermi, min_band, max_band) = initio.get_band_indices(scf_folder, energy_range = [-1.6, 1.6])
            print(f"{min_band = }, {max_band = }")
            
            chgcar_path = os.path.join(scf_folder, "CHGCAR")
            chgcar_destination = os.path.join(unfolding_folder, "CHGCAR")

            if os.path.isfile(chgcar_path): shutil.copy(chgcar_path, chgcar_destination)
            else: raise Exception("No CHGCAR found in the 'scf' folder")
        except Exception as e:
            print(f"Error encountered while running Initio.scf_to_unfolding: {e}")
        
        return



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

        def translate(self, vector: list = [0, 0, 0], frac_coords: bool = False) -> None:
            self.translate_sites(range(len(self.sites)), vector, frac_coords = frac_coords)
            return

        def flip(self, x: bool = False, y: bool = False, z: bool = True) -> None:
            for i, site in enumerate(self):
                new_coords = site.coords.copy()
                if x: new_coords[0] *= -1
                if y: new_coords[1] *= -1
                if z: new_coords[2] *= -1
                self[i] = (site.species, new_coords)
            return

        def to_molecule(self) -> Initio.Molecule:
            return Initio.Molecule(self.species, self.cart_coords)

    class Molecule(pmg_struct.Molecule):
        # Subclass of the pmg.core.Molecule class with convenience function exchange_atom
        def exchange_atom(self, index: int, element: str | int) -> None:
            if isinstance(element, int): # Convert from atomic number to element symbol
                elements = {el.Z: el.symbol for el in periodic_table.Element}
                element = elements[element]
            if not isinstance(index, int) or not isinstance(element, str): return
            
            self[index].species = element
            return

        def translate(self, vector: list = [0, 0, 0]) -> None:
            self.translate_sites(range(len(self.sites)), vector)
            return
        
        def translate_to_com(self, x: bool = True, y: bool = True, z: bool = True) -> None:
            com = self.center_of_mass
            vector = [0, 0, 0]
            if x: vector[0] = -com[0]
            if y: vector[1] = -com[1]
            if z: vector[2] = -com[2]
            
            self.translate(vector = vector)
            return

        def flip(self, x: bool = False, y: bool = False, z: bool = False) -> None:
            for i, site in enumerate(self):
                new_coords = site.coords.copy()
                if x: new_coords[0] *= -1
                if y: new_coords[1] *= -1
                if z: new_coords[2] *= -1
                self[i] = (site.species, new_coords)
            return

        def rotate(self, vector: list = [0, 0, 1], theta_deg: float = 0.) -> None:
            self.rotate_sites(range(len(self.sites)), theta = np.deg2rad(theta_deg), axis = vector)
            return


