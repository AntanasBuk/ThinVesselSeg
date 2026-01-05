import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, label
import argparse
import os
import multiprocessing as mp
from functools import partial
from utils import save_mask_as_image, merge_thin_and_thick_masks, open_file_as_binary_mask

def create_round_structuring_element(radius):
    y, x = np.ogrid[-radius:radius, -radius:radius]
    struct_elem = x**2 + y**2 <= radius**2
    return struct_elem


def custom_top_hat(mask, radius):
    struct_elem = create_round_structuring_element(radius + 1)
    eroded_mask = binary_erosion(mask, structure=struct_elem)

    struct_elem = create_round_structuring_element(radius + 2)
    opened_mask = binary_dilation(eroded_mask, structure=struct_elem)

    return opened_mask


def remove_small_components(mask, min_size):
    struct_elem = np.array([[1,1,1], [1,1,1], [1,1,1]], dtype=bool)
    labeled_mask, num_features = label(mask, struct_elem)
    for i in range(1, num_features + 1):
        component = labeled_mask == i
        coords = np.argwhere(component)
        
        if len(coords) > 0:
            min_row, min_col = coords.min(axis=0)
            max_row, max_col = coords.max(axis=0)
            
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            
            if height <= min_size  and width <= min_size:
                mask[component] = False

    return mask


def vessel_separation(input_path, thickness=8):
    vessel_mask = open_file_as_binary_mask(input_path)

    upscaled_vessel_mask = np.kron(vessel_mask, np.ones((2, 2), dtype=bool))
    upscaled_thick_vessel_mask = custom_top_hat(upscaled_vessel_mask, thickness)
    thick_vessel_mask = upscaled_thick_vessel_mask[::2, ::2]
    
    thin_vessel_mask = vessel_mask & ~thick_vessel_mask
    thin_vessel_mask = remove_small_components(thin_vessel_mask, min_size=thickness+2)

    thick_vessel_mask = vessel_mask & ~thin_vessel_mask
    thick_vessel_mask = remove_small_components(thick_vessel_mask, min_size=thickness+2)
    thin_vessel_mask = vessel_mask & ~thick_vessel_mask

    return thin_vessel_mask, thick_vessel_mask


def process_single_file(filename, input_folder, output_folder, thickness=8):
    input_file = os.path.join(input_folder, filename)
    output_file = os.path.join(output_folder, filename)

    thin_vessel_mask, thick_vessel_mask = vessel_separation(input_file, thickness)

    merged_mask = merge_thin_and_thick_masks(thin_vessel_mask, thick_vessel_mask)
    save_mask_as_image(merged_mask, output_file)


def distributed_vessel_separation(input_folder, output_folder, thickness=8, num_workers=None):
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png'))]

    if num_workers is None:
        num_workers = mp.cpu_count()

    worker_func = partial(process_single_file, 
                         input_folder=input_folder, 
                         output_folder=output_folder, 
                         thickness=thickness)

    with mp.Pool(num_workers) as pool:
        pool.map(worker_func, image_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract thin vessels from vessel masks')
    parser.add_argument('input', help='Input file or folder path')
    parser.add_argument('output', help='Output file or folder path')
    parser.add_argument('--thickness', type=int, default=8, help='Thickness parameter (default: 8)')
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers (default: all available cores)')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        output_path = args.output if not os.path.isdir(args.output) else os.path.join(args.output, os.path.basename(args.input))
        vessel_separation(args.input, output_path, args.thickness)

    elif os.path.isdir(args.input):
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        distributed_vessel_separation(args.input, args.output, args.thickness, args.workers)
        
    else:
        print(f"Error: Input path '{args.input}' does not exist")
