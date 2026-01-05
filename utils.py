from PIL import Image
import numpy as np

def extract_thin_and_thick_masks_from_image(img, thin_color=[255, 0, 0], thick_color=[255, 255, 255]):
    img_array = np.array(img)
    
    thin_mask = np.all(img_array == thin_color, axis=2)
    thick_mask = np.all(img_array == thick_color, axis=2)
    
    return thin_mask, thick_mask


def open_file_as_binary_mask(file_path):
    img = Image.open(file_path).convert('L')
    mask = np.array(img) > 127

    return mask


def merge_thin_and_thick_masks(thin_vessel_mask, thick_vessel_mask, thin_color=[255, 0, 0], thick_color=[255, 255, 255]):
    height, width = thick_vessel_mask.shape
    merged_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    merged_mask[thick_vessel_mask] = thick_color
    merged_mask[thin_vessel_mask] = thin_color
    
    return merged_mask


def save_mask_as_image(mask, output_path):
    if len(mask.shape) == 2:
        img = Image.fromarray((mask * 255).astype(np.uint8))
    else:
        img = Image.fromarray(mask.astype(np.uint8))
        
    img.save(output_path)