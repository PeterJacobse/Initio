import numpy as np
from pymatgen.core import structure as pmg_struct
from pymatgen.core import periodic_table
from typing import Literal, Self



# Structure manipulation
def combine_structures(structure1: Molecule | pmg_struct.Molecule | Structure | pmg_struct.Structure, structure2: Molecule | pmg_struct.Molecule | Structure | pmg_struct.Structure, translate1: list = [0, 0, 0], translate2: list = [0, 0, 0]) -> Molecule:
    """
    Combine two structures into a new structure. The output is a Molecule object since the periodicity may be ambiguous when combining two different structures with different lattices.

    Args:
        structure1 (Initio.Molecule | pmg_struct.Molecule | Initio.Structure | pmg_struct.Structure): Structure 1.
        structure2 (Initio.Molecule | pmg_struct.Molecule | Initio.Structure | pmg_struct.Structure): Structure 2.
        translate1 (list, optional): Translate the sites of structure 1 by this amount before combining. Defaults to [0, 0, 0].
        translate2 (list, optional): Translate the sites of structure 2 by this amount before combining. Defaults to [0, 0, 0].

    Returns:
        Initio.Molecule: Molecule that combines structure 1 and structure 2.
    """
    if not isinstance(structure1, Molecule | pmg_struct.Molecule | Structure | pmg_struct.Structure):
        raise Exception(f"Unsupported type for structure 1: {type(structure1)}")
    if not isinstance(structure2, Molecule | pmg_struct.Molecule | Structure | pmg_struct.Structure):
        raise Exception(f"Unsupported type for structure 2: {type(structure2)}")
    
    struc1_copy = structure1.copy()
    struc2_copy = structure2.copy()
    if np.sum(np.abs(translate1)) > .0001: struc1_copy.translate_sites(range(len(struc1_copy)), translate1)
    if np.sum(np.abs(translate2)) > .0001: struc2_copy.translate_sites(range(len(struc2_copy)), translate2)
    
    all_species = list(struc1_copy.species) + list(struc2_copy.species)
    all_coords = list(struc1_copy.cart_coords) + list(struc2_copy.cart_coords)
    return Molecule(all_species, all_coords)

def make_polyhedron(structure: Molecule | pmg_struct.Molecule | Structure | pmg_struct.Structure, sides: int = 5, size: int | float = 10., center_xy: list = [0, 0], start_angle_deg: int | float = 0.) -> Molecule:
    """
    Generate a regular polyhedral flake from a structure.

    Args:
        structure (Initio.Molecule | pmg_struct.Molecule | Initio.Structure | pmg_struct.Structure): Structure.
        sides (int, optional): Number of edges/vertices. Defaults to 5.
        size (int | float, optional): Size of the insribed circle. Defaults to 10..
        center_xy (list, optional): Center. Defaults to [0, 0].
        start_angle_deg (int | float, optional): Angle of the first edge relative to the structure. Defaults to 0.

    Returns:
        Initio.Molecule: Flake.
    """
    if not isinstance(structure, Molecule | pmg_struct.Molecule | Structure | pmg_struct.Structure):
        raise Exception(f"Unsupported type for structure: {type(structure)}")
    
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
    
    return Molecule(species = new_species, coords = new_coords)



class Structure(pmg_struct.Structure):
    """Subclass of the pmg.core.Structure class with convenience functions added.
    """
    def __init__(self, lattice, species, coords, **kwargs):
        super().__init__(lattice = lattice, species = species, coords = coords, **kwargs)

    @classmethod
    def from_file(cls, filename: str, primitive: bool = False, sort: bool = False, merge_tol: float = 0, **kwargs) -> Self:
        parent_instance = super().from_file(filename, primitive, sort, merge_tol, **kwargs)
        child_instance = cls.__new__(cls)
        [setattr(child_instance, key, value) for key, value in vars(parent_instance).items()]
        return child_instance

    @classmethod
    def GNR(cls, N: int = 2, orientation: Literal["a", "armchair", "z", "zigzag"] = "zigzag", n_supercell: int = 1, unit_cell_height_A: float | int = 10) -> Self:
        """Convenienve class method to construct a graphene nanoribbon.

        Args:
            N (int, optional): Width of the nanoribbon. Defaults to 2.
            orientation (str, optional): Orientation of the nanoribbon. Defaults to "armchair".
            n_supercell (int, optional): How many copies to generate in the periodic direction. Defaults to 1 (primitive cell).
            unit_cell_height_A (float | int, optional): The size of the vacuum region in the z direction between periodic images, expressed in Angstrom. Defaults to 10.

        Returns:
            initio.Structure: A Structure object
        """
        match orientation.lower():
            case "a": orientation = "a"
            case "armchair": orientation = "a"
            case "z": orientation = "z"
            case "zigzag": orientation = "z"
            case _: raise Exception(f"Unrecognized orientation {orientation} for GNR")
        
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
        lattice = pmg_struct.Lattice.from_parameters(latvec_x, latvec_y, latvec_z, 90, 90, 90)
        structure = cls(lattice = lattice, species = atomlist, coords = coords, coords_are_cartesian = True)
        if isinstance(n_supercell, int): structure.make_supercell([n_supercell, 1, 1])
        return structure
        
    def exchange_atom(self, index: int, element: str | int) -> None:
        """Function to substitute an atom in the Structure. Changes the coordinates in place.

        Args:
            index (int): The index of the atom in the Molecule object
            element (str | int): The element (symbol or index Z) to chage the atom into.
        """
        if isinstance(element, int): # Convert from atomic number to element symbol
            elements = {el.Z: el.symbol for el in periodic_table.Element} # type: ignore
            element = elements[element]
        if not isinstance(index, int) or not isinstance(element, str): return
        
        self[index].species = element # type: ignore
        return

    def translate(self, vector: list | np.ndarray = [0, 0, 0], frac_coords: bool = False) -> None:
        """Translates the Structure in the direction of the provided vector. Changes the coordinates in place.

        Args:
            vector (list | np.ndarray, optional): translation vector. Defaults to [0, 0, 0].
            frac_coords (bool, optional): Whether the vector is expressed in fractional coordinates of the structure or in Angstrom. Defaults to False.
        """
        self.translate_sites(range(len(self.sites)), list(vector), frac_coords = frac_coords)
        return

    def flip(self, x: bool = False, y: bool = False, z: bool = True) -> None:
        """Flips the coordinates of a Structure in the x, y, or z dimension. Changes the coordinates in place.

        Args:
            x (bool, optional): Whether to flip the x coordinates. Defaults to False.
            y (bool, optional): Whether to flip the y coordinates. Defaults to False.
            z (bool, optional): Whether to flip the z coordinates. Defaults to True.
        """
        for i, site in enumerate(self):
            new_coords = site.coords.copy()
            if x: new_coords[0] *= -1
            if y: new_coords[1] *= -1
            if z: new_coords[2] *= -1
            self[i] = (site.species, new_coords)
        return

    def to_molecule(self) -> Molecule:
        """Converts a Structure into a Molecule (discarding its lattice/periodicity properties).

        Returns:
            Initio.Molecule: The New Molecule object
        """
        return Molecule(self.species, self.cart_coords)



class Molecule(pmg_struct.Molecule):
    """Subclass of the pmg.core.Molecule class with convenience functions added.
    """
    def __init__(self, species, coords):
        super().__init__(species = species, coords = coords)

    def __add__(self, other):
        if not isinstance(other, Molecule | Structure): return Molecule(self.species, self.cart_coords) #
        combined_species = self.species + other.species
        combined_coords = np.vstack((self.cart_coords, other.cart_coords))        
        return Molecule(species = combined_species, coords = combined_coords)

    def exchange_atom(self, index: int, element: str | int) -> None:
        """Function to substitute an atom in the Molecule. Changes the coordinates in place.

        Args:
            index (int): The index of the atom in the Molecule object
            element (str | int): The element (symbol or index Z) to chage the atom into.
        """
        if isinstance(element, int): # Convert from atomic number to element symbol
            elements = {el.Z: el.symbol for el in periodic_table.Element} # type: ignore
            element = elements[element]
        if not isinstance(index, int) or not isinstance(element, str): return
        
        self[index].species = element # type: ignore
        return

    def translate(self, vector: list = [0, 0, 0]) -> None:
        """Translates the Molecule in the direction of the provided vector. Changes the coordinates in place.

        Args:
            vector (list | np.ndarray, optional): translation vector. Defaults to [0, 0, 0].
            frac_coords (bool, optional): Whether the vector is expressed in fractional coordinates of the structure or in Angstrom. Defaults to False.
        """
        self.translate_sites(range(len(self.sites)), vector)
        return
    
    def translate_to_com(self, x: bool = True, y: bool = True, z: bool = True) -> None:
        """Translates the x, y, or z coordinates of the Molecule to its center of mass in the corresponding direction. Changes the coordinates in place.

        Args:
            x (bool, optional): Whether to flip the x coordinates. Defaults to False.
            y (bool, optional): Whether to flip the y coordinates. Defaults to False.
            z (bool, optional): Whether to flip the z coordinates. Defaults to True.
        """
        com = self.center_of_mass
        vector = [0, 0, 0]
        if x: vector[0] = -com[0]
        if y: vector[1] = -com[1]
        if z: vector[2] = -com[2]
        
        self.translate(vector = vector)
        return

    def flip(self, x: bool = False, y: bool = False, z: bool = False) -> None:
        """Flips the coordinates of a Molecule in the x, y, or z dimension. Changes the coordinates in place.

        Args:
            x (bool, optional): Whether to flip the x coordinates. Defaults to False.
            y (bool, optional): Whether to flip the y coordinates. Defaults to False.
            z (bool, optional): Whether to flip the z coordinates. Defaults to True.
        """
        for i, site in enumerate(self):
            new_coords = site.coords.copy()
            if x: new_coords[0] *= -1
            if y: new_coords[1] *= -1
            if z: new_coords[2] *= -1
            self[i] = (site.species, new_coords)
        return

    def rotate(self, vector: list = [0, 0, 1], theta_deg: float = 0.) -> None:
        """Rotates a Molecule around a vector attached to the origin by an angle theta. Changes the coordinates in place.

        Args:
            x (bool, optional): Whether to flip the x coordinates. Defaults to False.
            y (bool, optional): Whether to flip the y coordinates. Defaults to False.
            z (bool, optional): Whether to flip the z coordinates. Defaults to True.
        """
        self.rotate_sites(range(len(self.sites)), theta = np.deg2rad(theta_deg), axis = vector)
        return

