import os, time, logging, threading
import nglview as nv
from PIL import Image
from .VaspBandUnfolding import vaspwfc
from pymatgen.io import vasp
from .structures import Structure
#from .unfold import UnfoldProcar



def find_folder(target: str = "", base_folder = "C:\\DFT") -> str:
    """Perform a nested search for a given folder under a base folder.

    Args:
        target (str, optional): Folder name. Defaults to "".
        base_folder (str, optional): Base folder path. Defaults to "C:\\DFT".

    Returns:
        str: path
    """
    target_folder = None
    sub_folders = [os.path.join(base_folder, folder) for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
    for sub_folder in sub_folders:
        if os.path.isdir(os.path.join(sub_folder, target)): target_folder = os.path.join(sub_folder, target)
    if not target_folder:
        for sub_folder in sub_folders:
            subsub_folders = [os.path.join(sub_folder, folder) for folder in os.listdir(sub_folder) if os.path.isdir(os.path.join(sub_folder, folder))]
            for subsub_folder in subsub_folders:
                if os.path.isdir(os.path.join(subsub_folder, target)): target_folder = os.path.join(subsub_folder, target)
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

def save_image(view: nv.NGLWidget, file_path: str = "") -> None:
    def save_image_when_ready(img_widget, file_path: str = "") -> None:
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



# VASP
def get_all_vasp_files(path: str) -> dict[str, object]:
    if os.path.isfile(path): folder = os.path.dirname(path)
    else: folder = path
    if not os.path.isdir(folder): raise FileNotFoundError("Invalid path provided")
    
    output_dict: dict[str, object] = {}
    files = [os.path.join(path, filename) for filename in os.listdir(folder)]

    for file in files:
        try:
            match os.path.basename(file):
                case "INCAR":
                    incar = get_incar(file)
                    output_dict.update({"INCAR": incar})
                case "CONTCAR" | "POSCAR":
                    struct = get_structure(file)
                    output_dict.update({"POSCAR": struct})
                case "WAVECAR":
                    wavecar = get_wavecar(file)
                    output_dict.update({"WAVECAR": wavecar})
                case "POTCAR":
                    potcar = get_potcar(file)
                    output_dict.update({"POTCAR": potcar})
                case "KPOINTS":
                    kpoints = get_kpoints(file)
                    output_dict.update({"KPOINTS": kpoints})
                case "EIGENVAL":
                    eigenval = get_eigenval(file)
                    output_dict.update({"EIGENVAL": eigenval})
                case "OUTCAR":
                    outcar = get_outcar(path)
                    output_dict.update({"OUTCAR": outcar})
                case _:
                    continue
        except Exception as e:
            print(f"{e}")
    return output_dict

def read_vasp_file(path: str) -> object:
    base_name = os.path.basename(path)
    
    match base_name:
        case "WAVECAR":
            try: return get_wavecar(path)
            except: pass
        case "POSCAR" | "CONTCAR":
            try: return get_structure(path)
            except: pass
        #case "PROCAR": return get_procar(path)
        case "INCAR":
            try: return get_incar(path)
            except: pass
        case "POTCAR":
            try: return get_potcar(path)
            except: pass
        case "KPOINTS":
            try: return get_kpoints(path)
            except: pass
        case "EIGENVAL":
            try: return get_eigenval(path)
            except: pass
        case "OUTCAR":
            try: return get_outcar(path)
            except: pass
        case _: raise Exception(f"Could not determine file type of provided file {path}")

def get_wavecar(path: str) -> vaspwfc:
    try:
        wfc = vaspwfc(path, lgamma = False)
        
        n_kpts = int(wfc._nkpts)
        if n_kpts < 2: wfc = vaspwfc(path, lgamma = True)
        
        return wfc
    except Exception as e:
        print("Error loading the wavecar")
        return vaspwfc()

def get_structure(path: str) -> Structure:
    try:
        structure = Structure.from_file(path)
        return structure
    except Exception as e:
        raise Exception(f"Could not open POSCAR / CONTCAR file: {e}")

def get_incar(path: str) -> vasp.inputs.Incar:
    try:
        incar = vasp.Incar.from_file(path)
        return incar
    except Exception as e:
        raise Exception(f"Could not opten INCAR file: {e}")

def get_potcar(path: str) -> vasp.inputs.Potcar:
    try:
        potcar = vasp.Potcar.from_file(path)
        return potcar
    except Exception as e:
        raise Exception(f"Could not open POTCAR file: {e}")

#def get_procar(path: str) -> UnfoldProcar:
#    try:
#        procar = UnfoldProcar(path)
#        return procar
#    except Exception as e:
#        raise Exception(f"Could not open PROCAR file: {e}")

def get_kpoints(path: str) -> vasp.inputs.Kpoints:
    try:
        kpoints = vasp.Kpoints.from_file(path)
        return kpoints
    except Exception as e:
        raise Exception(f"Could not open KPOINTS file: {e}")

def get_eigenval(path: str) -> vasp.outputs.Eigenval:
    try:
        eigenval = vasp.outputs.Eigenval(path)
        return eigenval
    except Exception as e:
        raise Exception(f"Could not open EIGENVAL file: {e}")

def get_outcar(path: str) -> vasp.outputs.Outcar:
    try:
        outcar = vasp.outputs.Outcar(path)
        return outcar
    except Exception as e:
        raise Exception(f"Could not open OUTCAR file: {e}")

def show_incar(incar: vasp.inputs.Incar) -> None:
    try:
        print(incar.get_str(pretty = True))
        return
    except Exception as e:
        raise Exception(f"Could not show Incar contents: {e}")


