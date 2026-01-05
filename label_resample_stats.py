import os
import argparse
from PIL import Image
import numpy as np
import multiprocessing as mp
from functools import partial
import cv2
from utils import merge_thin_and_thick_masks, extract_thin_and_thick_masks_from_image


def maxpool(thin_vessel_mask, thick_vessel_mask, target_size):
    pool_size = (thin_vessel_mask.shape[0] // target_size[1], thin_vessel_mask.shape[1] // target_size[0])

    thin_resized = np.zeros((target_size[1], target_size[0]), dtype=bool)
    thick_resized = np.zeros((target_size[1], target_size[0]), dtype=bool)
    
    for i in range(target_size[1]):
        for j in range(target_size[0]):
            y_start = i * pool_size[0]
            y_end = min((i + 1) * pool_size[0], thin_vessel_mask.shape[0])
            x_start = j * pool_size[1]
            x_end = min((j + 1) * pool_size[1], thin_vessel_mask.shape[1])
            
            thin_resized[i, j] = np.max(thin_vessel_mask[y_start:y_end, x_start:x_end])
            thick_resized[i, j] = np.max(thick_vessel_mask[y_start:y_end, x_start:x_end])

    merged_mask = merge_thin_and_thick_masks(thin_resized, thick_resized)

    return Image.fromarray(merged_mask.astype(np.uint8))
            

def resize_bilinear(thin_vessel_mask, thick_vessel_mask, target_size, threshold=127):
    thin_vessel_mask_resized = cv2.resize(thin_vessel_mask.astype(np.uint8) * 255, target_size, interpolation=cv2.INTER_LINEAR)
    thick_vessel_mask_resized = cv2.resize(thick_vessel_mask.astype(np.uint8) * 255, target_size, interpolation=cv2.INTER_LINEAR)
        
    thin_binary = thin_vessel_mask_resized > threshold
    thick_binary = thick_vessel_mask_resized > threshold
        
    merged_mask = merge_thin_and_thick_masks(thin_binary, thick_binary)

    return Image.fromarray(merged_mask.astype(np.uint8))


def resize_bicubic(thin_vessel_mask, thick_vessel_mask, target_size, threshold=127):
    thin_vessel_mask_resized = cv2.resize(thin_vessel_mask.astype(np.uint8) * 255, target_size, interpolation=cv2.INTER_CUBIC)
    thick_vessel_mask_resized = cv2.resize(thick_vessel_mask.astype(np.uint8) * 255, target_size, interpolation=cv2.INTER_CUBIC)
        
    thin_binary = thin_vessel_mask_resized > threshold
    thick_binary = thick_vessel_mask_resized > threshold
        
    merged_mask = merge_thin_and_thick_masks(thin_binary, thick_binary)

    return Image.fromarray(merged_mask.astype(np.uint8))


def distance_transform(thin_vessel_mask, thick_vessel_mask, target_size, threshold=127):
    dist_transform_thin = cv2.distanceTransform(thin_vessel_mask.astype(np.uint8), cv2.DIST_L2, 5)
    resized_dist_thin = cv2.resize(dist_transform_thin, target_size, interpolation=cv2.INTER_LINEAR)
    threshold_size = threshold / 255.0
    thin_binary = (resized_dist_thin > threshold_size).astype(bool)

    dist_transform_thick = cv2.distanceTransform(thick_vessel_mask.astype(np.uint8), cv2.DIST_L2, 5)
    resized_dist_thick = cv2.resize(dist_transform_thick, target_size, interpolation=cv2.INTER_LINEAR)
    thick_binary = (resized_dist_thick > threshold_size).astype(bool)

    merged_mask = merge_thin_and_thick_masks(thin_binary, thick_binary)

    return Image.fromarray(merged_mask.astype(np.uint8))
            

def calculate_stats(gt_image, alt_image, vessel_color = np.array([255, 0, 0])):
    if gt_image.shape != alt_image.shape:
        raise ValueError(f"Ground truth and altered images must have the same dimensions ({gt_image.shape} vs {alt_image.shape})")
     
    background_mask = np.all(gt_image == np.array([0, 0, 0]), axis=-1)
    ground_truth_mask = np.all(gt_image == vessel_color, axis=-1)
    prediction_mask = np.all(alt_image == vessel_color, axis=-1)
    
    TP = np.sum(np.logical_and(prediction_mask, ~background_mask))
    FP = np.sum(np.logical_and(prediction_mask, background_mask))
    FN = np.sum(np.logical_and(~prediction_mask, ground_truth_mask))
     
    stats = {
        'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
        'recall': TP / (TP + FN) if (TP + FN) > 0 else 0,
        'f1_score': (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0,
    }
    
    return stats
    
def resize_stats(file_path, target_size):
    stats = {}

    with Image.open(file_path) as img:
        original_size = img.size

        thin_vessel_mask, thick_vessel_mask = extract_thin_and_thick_masks_from_image(img)

        downsampled_img = img.resize(target_size, Image.NEAREST)
        upsampled_img = downsampled_img.resize(original_size, Image.NEAREST)
        thin_vessel_stats = calculate_stats(np.array(img), np.array(upsampled_img), np.array([255,0,0]))
        thick_vessel_stats = calculate_stats(np.array(img), np.array(upsampled_img), np.array([255,255,255]))
        stats['nearest_neighbor'] = (thin_vessel_stats, thick_vessel_stats)

        downsampled_img = maxpool(thin_vessel_mask, thick_vessel_mask, target_size)
        upsampled_img = downsampled_img.resize(original_size, Image.NEAREST)
        thin_vessel_stats = calculate_stats(np.array(img), np.array(upsampled_img), np.array([255,0,0]))
        thick_vessel_stats = calculate_stats(np.array(img), np.array(upsampled_img), np.array([255,255,255]))
        stats['maxpool'] = (thin_vessel_stats, thick_vessel_stats)

        downsampled_img = resize_bilinear(thin_vessel_mask, thick_vessel_mask, target_size, 96)
        upsampled_img = downsampled_img.resize(original_size, Image.NEAREST)
        thin_vessel_stats = calculate_stats(np.array(img), np.array(upsampled_img), np.array([255,0,0]))
        thick_vessel_stats = calculate_stats(np.array(img), np.array(upsampled_img), np.array([255,255,255]))
        stats['bilinear'] = (thin_vessel_stats, thick_vessel_stats)

        downsampled_img = resize_bicubic(thin_vessel_mask, thick_vessel_mask, target_size, 98)
        upsampled_img = downsampled_img.resize(original_size, Image.NEAREST)
        thin_vessel_stats = calculate_stats(np.array(img), np.array(upsampled_img), np.array([255,0,0]))
        thick_vessel_stats = calculate_stats(np.array(img), np.array(upsampled_img), np.array([255,255,255]))
        stats['bicubic'] = (thin_vessel_stats, thick_vessel_stats)

        downsampled_img = distance_transform(thin_vessel_mask, thick_vessel_mask, target_size, 96)
        upsampled_img = downsampled_img.resize(original_size, Image.NEAREST)
        thin_vessel_stats = calculate_stats(np.array(img), np.array(upsampled_img), np.array([255,0,0]))
        thick_vessel_stats = calculate_stats(np.array(img), np.array(upsampled_img), np.array([255,255,255]))
        stats['distance_transform'] = (thin_vessel_stats, thick_vessel_stats)

        return file_path, stats

def print_vessel_stats(vessel_class, stats_list):
    if stats_list:
        f1_scores = [stats[0]['f1_score'] for stats in stats_list]
        precision_scores = [stats[0]['precision'] for stats in stats_list]
        recall_scores = [stats[0]['recall'] for stats in stats_list]
        
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        
        worst_idx = np.argmin(f1_scores)
        worst_f1 = f1_scores[worst_idx]
        worst_path = stats_list[worst_idx][1]
        
        print(f"{vessel_class} Vessels:")
        print(f"  Average Precision: {avg_precision:.4f} ± {np.std(precision_scores):.4f}")
        print(f"  Average Recall: {avg_recall:.4f} ± {np.std(recall_scores):.4f}")
        print(f"  Average F1-Score: {avg_f1:.4f} ± {np.std(f1_scores):.4f}")
        print(f"  Worst F1-Score: {worst_f1:.4f} - {worst_path}")

def print_results(results):
    method_stats = {}
    for result in results:
        if result is not None and isinstance(result, tuple) and len(result) == 2:
            file_path, stats = result
            
            for method, (thin_vessel_stats, thick_vessel_stats) in stats.items():
                if method not in method_stats:
                    method_stats[method] = {'thin': [], 'thick': []}
                method_stats[method]['thin'].append((thin_vessel_stats, file_path))
                method_stats[method]['thick'].append((thick_vessel_stats, file_path))

    for method_name, vessel_stats in method_stats.items():
        print(f"\n=== {method_name.replace('_', ' ').title()} Statistics for {target_size[0]}x{target_size[1]} ===")
        print_vessel_stats("Thin", vessel_stats['thin'])
        print_vessel_stats("Thick", vessel_stats['thick'])

def process_single_file(filename, input_folder, target_size):
    input_file = os.path.join(input_folder, filename)

    return resize_stats(input_file, target_size)


def distributed_resize_stats(input_folder, target_size, num_workers=None):
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png'))]
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    worker_func = partial(process_single_file, 
                         input_folder=input_folder, 
                         target_size=target_size)
    
    with mp.Pool(num_workers) as pool:
        results = pool.map(worker_func, image_files)
    
    return results
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize images preserving folder structure')
    parser.add_argument('input', help='Input file or folder path')
    parser.add_argument('--width', type=int, default=1024, help='Target width (default: 1024)')
    parser.add_argument('--height', type=int, default=1024, help='Target height (default: 1024)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: CPU count)')
    
    args = parser.parse_args()
    
    target_size = (args.width, args.height)
    
    if os.path.isfile(args.input):
        result = resize_stats(args.input, target_size)
        print_results([result])

    elif os.path.isdir(args.input):
        results = distributed_resize_stats(args.input, target_size, args.workers)
        print_results(results)
