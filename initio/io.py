import os, time, logging, threading
import nglview as nv
from PIL import Image



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
