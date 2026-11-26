import os
import json
import numpy as np
from PIL import Image
from flask import Flask, send_file, jsonify, request
import io
import argparse
import base64
from datetime import datetime
import time
import glob
import uuid
import threading
import re
import shutil
import colorsys
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------
# ARGPARSE
# ------------------------------------------------

ROOT = 'datasets'
SIZE = 512
HOST = '0.0.0.0'
PORT = 5000
DEBUG = False
MAX_IMAGES = 10
WORKERS = 8

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default='datasets_to_review')
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--max_images", type=int, default=10, help="Maximum number of images to load per dataset")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for loading")
    return parser.parse_args()

# ------------------------------------------------
# UTIL
# ------------------------------------------------

def get_images(img_dir, max_images=10):
    """Get first N images, prioritizing numeric filenames"""
    if not img_dir or not os.path.exists(img_dir):
        return []
    
    all_files = [f for f in os.listdir(img_dir) 
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".nii", ".nii.gz"))]
    
    # Try to sort numerically if filenames are numbers
    def numeric_sort_key(filename):
        match = re.match(r'^(\d+)', filename)
        if match:
            return int(match.group(1))
        return float('inf')
    
    # Check if all files start with numbers
    numeric_files = [f for f in all_files if re.match(r'^\d+', f)]
    if len(numeric_files) == len(all_files):
        sorted_files = sorted(all_files, key=numeric_sort_key)
    else:
        sorted_files = sorted(all_files)
    
    # Return only the first max_images
    return sorted_files[:max_images]

def get_dirs(path):
    if not path or not os.path.exists(path):
        return []
    return sorted([d for d in os.listdir(path)
                   if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')])

def get_nifti_shape(path):
    """Get the shape of a NIfTI file without loading the entire volume"""
    try:
        img = nib.load(path)
        return img.shape
    except Exception as e:
        print(f"Error getting shape for {path}: {e}")
        return None

def load_nifti_slice(path, slice_idx):
    """Load only a specific slice from a NIfTI file"""
    try:
        img = nib.load(path)
        data = img.dataobj
        # Get the slice along the third dimension
        if len(data.shape) == 3:
            slice_data = data[:, :, slice_idx]
            # Convert to numpy array
            return np.array(slice_data)
        else:
            # If not 3D, try to get the first slice
            return np.array(data[:, :, 0])
    except Exception as e:
        print(f"Error loading slice from {path}: {e}")
        return None

def normalize_ct_image(image_data, window_center=40, window_width=400):
    """
    Normalize CT image using windowing
    """
    min_val = window_center - window_width // 2
    max_val = window_center + window_width // 2
    
    image_data = np.clip(image_data, min_val, max_val)
    
    if max_val > min_val:
        image_data = (image_data - min_val) / (max_val - min_val) * 255
    else:
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data) + 1e-8) * 255
    
    return image_data.astype(np.uint8)

def get_color_for_class(class_index, num_classes):
    """Generate distinct colors for each class"""
    if num_classes == 1:
        return (30, 144, 255)
    
    hue = class_index / num_classes
    hue = (hue + 0.1) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return tuple(int(c * 255) for c in rgb)



def process_single_image_fast(img_file, img_dir, label_paths_by_image, size, modality):
    """Process a single image - load only the middle slice"""
    try:
        img_path = os.path.join(img_dir, img_file)
        label_paths = label_paths_by_image.get(img_file, [])
        
        # Determine if this is a NIfTI file
        is_nifti = img_path.endswith(('.nii', '.nii.gz'))
        
        if is_nifti:
            # Get image shape to find middle slice
            img_shape = get_nifti_shape(img_path)
            if img_shape is None:
                raise Exception("Failed to get image shape")
            
            # Calculate middle slice
            if len(img_shape) == 3:
                slice_idx = img_shape[2] // 2
            else:
                slice_idx = 0
            
            # Load ONLY the middle slicef
            image_slice = load_nifti_slice(img_path, slice_idx)
            if image_slice is None:
                raise Exception("Failed to load image slice")
            
            # Normalize CT images for CT modality
            if modality == 'CT':
                image_slice = normalize_ct_image(image_slice)
            else:
                # For MRI, normalize differently
                if np.max(image_slice) > 0:
                    image_slice = (image_slice / np.max(image_slice) * 255).astype(np.uint8)
                else:
                    image_slice = image_slice.astype(np.uint8)

            
            # Resize
            if image_slice.shape[0] != size or image_slice.shape[1] != size:
                img_pil = Image.fromarray(image_slice)
                img_pil = img_pil.resize((size, size), Image.Resampling.LANCZOS)
                image_slice = np.array(img_pil)
            
            # Convert to RGB
            if len(image_slice.shape) == 2:
                image_slice = np.stack([image_slice]*3, axis=2)

            
            img_pil = Image.fromarray(image_slice)
            
            # Load label slices (only the middle slice)
            colored_mask = create_colored_mask_from_labels(label_paths, slice_idx, size)
            
            return img_pil, colored_mask, slice_idx, len(label_paths)
        
        else:
            # Handle 2D image
            img = Image.open(img_path).convert("RGB")
            img_pil = img.resize((size, size))
            
            # Create colored mask for 2D
            colored_mask = create_colored_mask_from_labels_2d(label_paths, size)
            
            return img_pil, colored_mask, 0, len(label_paths)
            
    except Exception as e:
        print(f"Error processing {img_file}: {e}")
        # Return empty image
        empty_img = Image.new("RGB", (size, size), (100, 100, 100))
        empty_mask = Image.new("RGB", (size, size), (0, 0, 0))
        return empty_img, empty_mask, 0, 0

def create_colored_mask_from_labels(label_paths, slice_idx, size):
    """Create colored mask from label slices - only loads the needed slice"""
    colored_mask = np.zeros((size, size, 3), dtype=np.uint8)
    
    if not label_paths:
        return Image.fromarray(colored_mask)
    
    # Group labels by class
    class_labels = {}
    for label_path in label_paths:
        class_name = os.path.basename(os.path.dirname(label_path))
        if not class_name or class_name == '.':
            class_name = "mask"
        
        if class_name not in class_labels:
            class_labels[class_name] = []
        class_labels[class_name].append(label_path)
    
    class_names_sorted = sorted(class_labels.keys())
    num_classes = len(class_names_sorted)
    
    for i, class_name in enumerate(class_names_sorted):
        color = get_color_for_class(i, num_classes)
        
        for label_path in class_labels[class_name]:
            try:
                # Load ONLY the middle slice from the label
                slice_data = load_nifti_slice(label_path, slice_idx)
                if slice_data is None:
                    continue
                
                # Resize if needed using nearest neighbor (fast)
                if slice_data.shape[0] != size or slice_data.shape[1] != size:
                    from scipy.ndimage import zoom
                    zoom_factor = size / slice_data.shape[0]
                    slice_data = zoom(slice_data, (zoom_factor, zoom_factor), order=0)
                
                mask_array = (slice_data > 0).astype(np.uint8)
                
                # Apply color (vectorized)
                for c in range(3):
                    colored_mask[:, :, c] = np.where(
                        mask_array > 0,
                        color[c],
                        colored_mask[:, :, c]
                    )
            except Exception as e:
                print(f"Error processing mask {label_path}: {e}")
                continue
    
    return Image.fromarray(colored_mask)

def create_colored_mask_from_labels_2d(label_paths, size):
    """Create colored mask from 2D labels"""
    colored_mask = np.zeros((size, size, 3), dtype=np.uint8)
    
    if not label_paths:
        return Image.fromarray(colored_mask)
    
    # Group labels by class
    class_labels = {}
    for label_path in label_paths:
        class_name = os.path.basename(os.path.dirname(label_path))
        if not class_name or class_name == '.':
            class_name = "mask"
        
        if class_name not in class_labels:
            class_labels[class_name] = []
        class_labels[class_name].append(label_path)
    
    class_names_sorted = sorted(class_labels.keys())
    num_classes = len(class_names_sorted)
    
    for i, class_name in enumerate(class_names_sorted):
        color = get_color_for_class(i, num_classes)
        
        for label_path in class_labels[class_name]:
            try:
                # Load the 2D label
                img = Image.open(label_path).convert("L")
                img = img.resize((size, size), Image.Resampling.NEAREST)
                mask_array = (np.array(img) > 128).astype(np.uint8)
                
                # Apply color
                for c in range(3):
                    colored_mask[:, :, c] = np.where(
                        mask_array > 0,
                        color[c],
                        colored_mask[:, :, c]
                    )
            except Exception as e:
                print(f"Error processing mask {label_path}: {e}")
                continue
    
    return Image.fromarray(colored_mask)

# ------------------------------------------------
# GLOBAL STATE
# ------------------------------------------------

global_state = {
    "modality": None,
    "dataset": None,
    "index": 0,
    "images": [],
    "image_data": [],
    "mask_data": [],
    "image_names": [],
    "image_labels": [],
    "slice_info": [],
    "total": 0,
    "is_loading": False
}

state_lock = threading.Lock()

def find_image_directories(modality, dataset):
    """Find all possible image directories"""
    base_path = os.path.join(ROOT, modality, dataset)
    possible_dirs = [
        os.path.join(base_path, "converted", "images"),
        os.path.join(base_path, "images"),
        base_path,
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            return dir_path
    return None

def find_label_directories(modality, dataset):
    """Find all possible label directories"""
    base_path = os.path.join(ROOT, modality, dataset)
    possible_dirs = [
        os.path.join(base_path, "converted", "labels"),
        os.path.join(base_path, "labels"),
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            return dir_path
    return None

def find_label_paths(img_file, label_dir, label_subdirs):
    """Find label paths for an image file"""
    label_paths = []
    
    # Try different filename variations
    base_name = img_file
    extensions_to_try = ['', '.nii', '.nii.gz']
    
    # Also try without any extension
    base_no_ext = os.path.splitext(img_file)[0]
    base_no_ext2 = os.path.splitext(base_no_ext)[0]  # Handle .nii.gz
    
    names_to_try = [base_name, base_no_ext, base_no_ext2]
    
    if label_subdirs:
        # Labels are organized in class folders
        for label_name in label_subdirs:
            for name in names_to_try:
                for ext in extensions_to_try:
                    label_path = os.path.join(label_dir, label_name, name + ext)
                    if os.path.exists(label_path):
                        label_paths.append(label_path)
                        break
                if label_paths and label_paths[-1] in label_paths:
                    break
    else:
        # Labels directly in label directory
        for name in names_to_try:
            for ext in extensions_to_try:
                label_path = os.path.join(label_dir, name + ext)
                if os.path.exists(label_path):
                    label_paths.append(label_path)
                    break
            if label_paths:
                break
    
    return label_paths

def preload_images(modality, dataset):
    """Pre-load and cache all images for a dataset with parallel processing"""
    with state_lock:
        if global_state.get("is_loading", False):
            print("Already loading, please wait...")
            return 0
        
        global_state["is_loading"] = True
        
        img_dir = find_image_directories(modality, dataset)
        label_dir = find_label_directories(modality, dataset)
        
        if not img_dir:
            print(f"No image directory found for {modality}/{dataset}")
            global_state["is_loading"] = False
            return 0
        
        # Get first N images
        image_files = get_images(img_dir, MAX_IMAGES)
        
        if not image_files:
            print(f"No images found in {img_dir}")
            global_state["is_loading"] = False
            return 0
        
        print(f"Loading {len(image_files)} images from {modality}/{dataset}")
        
        # Get label subdirectories
        label_subdirs = []
        if label_dir and os.path.exists(label_dir):
            label_subdirs = get_dirs(label_dir)
            if label_subdirs:
                print(f"Found {len(label_subdirs)} label classes")
        
        # Build label paths for each image
        label_paths_by_image = {}
        for img_file in image_files:
            label_paths_by_image[img_file] = find_label_paths(img_file, label_dir, label_subdirs) if label_dir else []
        
        # Process images in parallel
        print(f"Processing with {WORKERS} workers (loading only middle slice)...")
        
        image_data_list = []
        mask_data_list = []
        image_names = []
        image_labels_list = []
        slice_info_list = []
        
        # Prepare arguments for parallel processing
        args_list = [
            (img_file, img_dir, label_paths_by_image, SIZE, modality)
            for img_file in image_files
        ]
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(process_single_image_fast, *arg) for arg in args_list]
            
            for i, future in enumerate(as_completed(futures)):
                img_file = image_files[i]
                try:
                    img_pil, colored_mask, slice_idx, num_labels = future.result()
                    
                    # Convert to base64
                    img_buffer = io.BytesIO()
                    img_pil.save(img_buffer, format="PNG")
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    image_data_list.append(img_base64)
                    image_names.append(img_file)
                    slice_info_list.append(slice_idx)
                    
                    mask_buffer = io.BytesIO()
                    colored_mask.save(mask_buffer, format="PNG")
                    mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
                    mask_data_list.append(mask_base64)
                    
                    # Get class names
                    class_names = []
                    label_paths = label_paths_by_image.get(img_file, [])
                    for label_path in label_paths:
                        class_name = os.path.basename(os.path.dirname(label_path))
                        if class_name and class_name not in class_names:
                            class_names.append(class_name)
                    image_labels_list.append(class_names)
                    
                    if num_labels > 0:
                        print(f"  {img_file} (slice {slice_idx})")
                    else:
                        print(f"  {img_file} (no labels)")
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    # Add empty entry
                    empty_img = Image.new("RGB", (SIZE, SIZE), (100, 100, 100))
                    empty_mask = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
                    img_buffer = io.BytesIO()
                    empty_img.save(img_buffer, format="PNG")
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    image_data_list.append(img_base64)
                    image_names.append(img_file)
                    slice_info_list.append(0)
                    mask_buffer = io.BytesIO()
                    empty_mask.save(mask_buffer, format="PNG")
                    mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
                    mask_data_list.append(mask_base64)
                    image_labels_list.append([])
        
        # Store in global state
        global_state["modality"] = modality
        global_state["dataset"] = dataset
        global_state["images"] = image_names
        global_state["image_data"] = image_data_list
        global_state["mask_data"] = mask_data_list
        global_state["image_labels"] = image_labels_list
        global_state["slice_info"] = slice_info_list
        global_state["index"] = 0
        global_state["total"] = len(image_names)
        global_state["is_loading"] = False
        
        print(f"Loaded {len(image_names)} images")
        return len(image_names)

# ------------------------------------------------
# APP
# ------------------------------------------------

def create_app(root='datasets', size=512, host='0.0.0.0', port=5000, debug=False):
    app = Flask(__name__)
    app.secret_key = 'browse-only-no-auth'
    
    @app.route("/")
    def index():
        return HTML_PAGE

    @app.route("/modalities")
    def modalities():
        return jsonify([el for el in get_dirs(root) if el in ['CT', 'MRI', 'Dermoscopy', 'Microscopy', 'Endoscopy', 'Fundus', 'OCT', 'Ultrasound', 'XRay']])

    @app.route("/set_modality")
    def set_modality():
        modality = request.args.get("name")
        if not modality:
            return "error", 400
        
        with state_lock:
            global_state["modality"] = modality
            global_state["dataset"] = None
            global_state["images"] = []
            global_state["image_data"] = []
            global_state["mask_data"] = []
            global_state["index"] = 0
            global_state["total"] = 0
        
        return "ok"

    @app.route("/datasets")
    def datasets():
        with state_lock:
            modality = global_state.get("modality")
        if not modality:
            return jsonify([])
        
        datasets = get_dirs(os.path.join(root, modality))
        return jsonify(datasets)

    @app.route("/set_dataset")
    def set_dataset():
        dataset = request.args.get("name")
        if not dataset:
            return "error", 400
        
        with state_lock:
            modality = global_state.get("modality")
            # Check if already loading
            if global_state.get("is_loading", False):
                return jsonify({"status": "loading", "message": "Dataset is already loading"}), 202
        
        if modality:
            # Run loading in background thread to not block
            def load_in_background():
                preload_images(modality, dataset)
            
            thread = threading.Thread(target=load_in_background)
            thread.daemon = True
            thread.start()
            
            # Return immediately with loading status
            return jsonify({"status": "loading", "message": "Loading dataset in background"}), 202
        
        return "ok"

    @app.route("/loading_status")
    def loading_status():
        with state_lock:
            is_loading = global_state.get("is_loading", False)
            total = global_state.get("total", 0)
            return jsonify({
                "is_loading": is_loading,
                "total": total,
                "loaded": len(global_state.get("images", []))
            })

    @app.route("/next")
    def next():
        with state_lock:
            if global_state["images"] and global_state["index"] < len(global_state["images"]) - 1:
                global_state["index"] += 1
        return "ok"

    @app.route("/back")
    def back():
        with state_lock:
            if global_state["index"] > 0:
                global_state["index"] -= 1
        return "ok"
    
    @app.route("/info")
    def info():
        with state_lock:
            total = global_state.get("total", 0)
            if total == 0:
                return jsonify({"done": True, "total": 0, "index": 0})
            
            index = global_state.get("index", 0)
            if index >= total:
                return jsonify({"done": True, "total": total, "index": index})
            
            current_name = global_state["images"][index]
            slice_idx = global_state["slice_info"][index] if index < len(global_state.get("slice_info", [])) else 0
            
            return jsonify({
                "done": False,
                "index": index,
                "total": total,
                "name": current_name,
                "slice": slice_idx
            })
        
    @app.route("/image_labels")
    def image_labels():
        with state_lock:
            index = global_state.get("index", 0)
            if index < len(global_state.get("image_labels", [])):
                return jsonify(global_state["image_labels"][index])
            return jsonify([])

    @app.route("/image_data")
    def image_data():
        with state_lock:
            index = global_state.get("index", 0)
            if index < len(global_state.get("image_data", [])):
                return jsonify({
                    "image": global_state["image_data"][index],
                    "mask": global_state["mask_data"][index]
                })
            return jsonify({"image": "", "mask": ""})

    @app.route("/reset")
    def reset():
        with state_lock:
            global_state["modality"] = None
            global_state["dataset"] = None
            global_state["images"] = []
            global_state["image_data"] = []
            global_state["mask_data"] = []
            global_state["index"] = 0
            global_state["total"] = 0
        return "ok"

    return app



# ------------------------------------------------
# HTML PAGE WITH COOL VISUAL ENHANCEMENTS
# ------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Browser</title>
    <style>
        /* ===== CSS VARIABLES FOR THEME ===== */
        :root {
            --bg-primary: #f0f2f5;
            --bg-secondary: #ffffff;
            --text-primary: #1a1a2e;
            --text-secondary: #666;
            --card-shadow: 0 2px 8px rgba(0,0,0,0.1);
            --border-color: #e2e8f0;
            --gradient-start: #667eea;
            --gradient-end: #764ba2;
            --glass-bg: rgba(255, 255, 255, 0.7);
        }

        [data-theme="dark"] {
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --card-shadow: 0 2px 8px rgba(0,0,0,0.3);
            --border-color: #2d2d44;
            --glass-bg: rgba(26, 26, 46, 0.7);
        }

        /* ===== GLOBAL STYLES ===== */
        * { box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 20px;
            transition: background 0.3s ease, color 0.3s ease;
            min-height: 100vh;
        }

        /* ===== PARTICLES BACKGROUND ===== */
        #particles-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            pointer-events: none;
        }

        /* ===== THEME TOGGLE ===== */
        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 50%;
            width: 44px;
            height: 44px;
            cursor: pointer;
            font-size: 20px;
            z-index: 1000;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: var(--card-shadow);
        }
        .theme-toggle:hover {
            transform: scale(1.1) rotate(10deg);
        }

        /* ===== CONTAINER ===== */
        .container { max-width: 1400px; margin: 0 auto; position: relative; z-index: 1; }

        /* ===== GLASS CARD ===== */
        .card {
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: var(--card-shadow);
            transition: all 0.3s ease;
        }
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        }
        [data-theme="dark"] .card {
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        [data-theme="dark"] .card:hover {
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }

        /* ===== GRADIENT BORDER CARD ===== */
        .card-gradient {
            position: relative;
            border: 2px solid transparent;
            background-clip: padding-box;
            color: white;
        }
        .card-gradient kbd {
            background: rgba(255,255,255,0.15);
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card-gradient::before {
            content: '';
            position: absolute;
            inset: -2px;
            border-radius: 16px;
            background: linear-gradient(45deg, var(--gradient-start), var(--gradient-end), #f093fb);
            z-index: -1;
            animation: border-rotate 4s linear infinite;
            background-size: 300% 300%;
        }
        @keyframes border-rotate {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* ===== MAIN LAYOUT ===== */
        .main-layout { display: flex; gap: 24px; flex-wrap: wrap; }
        .left-panel { flex: 2; min-width: 300px; }

        /* ===== CANVAS ===== */
        canvas {
            width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            background: #1e293b;
            display: block;
            transition: opacity 0.3s ease, transform 0.3s ease;
        }
        canvas.loading {
            opacity: 0.6;
            transform: scale(0.98);
        }
        canvas.loaded {
            opacity: 1;
            transform: scale(1);
        }

        /* ===== BUTTONS ===== */
        .controls { margin: 15px 0; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
        
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        button::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            transform: translate(-50%, -50%);
            transition: width 0.6s ease, height 0.6s ease;
        }
        button:hover::after {
            width: 300px;
            height: 300px;
        }
        button:hover { transform: translateY(-2px); filter: brightness(1.05); }
        button:active { transform: scale(0.95); }
        
        .btn-nav { background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end)); color: white; }
        .btn-secondary { background: var(--border-color); color: var(--text-primary); }
        .btn-secondary:hover { background: var(--text-secondary); color: white; }

        /* ===== SLIDER ===== */
        .opacity-slider { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
        .opacity-slider input[type="range"] {
            flex: 1;
            height: 6px;
            -webkit-appearance: none;
            background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
            border-radius: 3px;
            outline: none;
        }
        .opacity-slider input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: white;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }

        /* ===== BADGES ===== */
        .badge {
            display: inline-block;
            padding: 4px 12px;
            background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
            color: white;
            border-radius: 20px;
            font-size: 12px;
            margin: 2px;
            font-weight: 500;
        }
        .dataset-badge {
            background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
            color: white;
            padding: 8px 16px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .slice-badge {
            background: #f59e0b;
            color: white;
            padding: 2px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            animation: pulse-badge 2s ease-in-out infinite;
        }
        @keyframes pulse-badge {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .speed-badge {
            background: #10b981;
            color: white;
            padding: 2px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 10px;
        }

        /* ===== COLOR INDICATOR ===== */
        .color-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 4px;
            margin-right: 6px;
        }
        .class-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
            padding: 10px;
            background: var(--bg-primary);
            border-radius: 8px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            font-size: 13px;
        }

        /* ===== LOADING ===== */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            backdrop-filter: blur(10px);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            animation: fade-in 0.3s ease;
        }
        .loading-content {
            background: var(--bg-secondary);
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            animation: pulse 2s ease-in-out infinite;
        }
        .spinner {
            border: 4px solid rgba(255,255,255,0.1);
            border-top: 4px solid var(--gradient-start);
            border-radius: 50%;
            width: 60px;
            height: 60px;
            margin: 0 auto 20px;
            animation: spin 1s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg) scale(1); }
            50% { transform: rotate(180deg) scale(1.1); }
            100% { transform: rotate(360deg) scale(1); }
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        @keyframes fade-in {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }

        /* ===== SELECTORS ===== */
        .modal-selectors {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 10px 0;
        }
        .selector-btn {
            padding: 8px 18px;
            border: 2px solid var(--border-color);
            border-radius: 10px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        .selector-btn:hover { 
            border-color: var(--gradient-start);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        }
        .selector-btn.active { 
            border-color: var(--gradient-start);
            background: linear-gradient(135deg, var(--gradient-start), var(--gradient-end));
            color: white;
        }

        /* ===== STATS ===== */
        .stats {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin: 10px 0;
        }
        .stat-item {
            background: var(--bg-primary);
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
        }
        .stat-item strong { color: var(--gradient-start); }

        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-primary);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, var(--gradient-start), var(--gradient-end));
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(45deg, var(--gradient-end), var(--gradient-start));
        }

        /* ===== RESPONSIVE ===== */
        @media (max-width: 768px) {
            body { padding: 10px; }
            .main-layout { flex-direction: column; }
            .controls button { flex: 1; min-width: 80px; font-size: 12px; padding: 8px 12px; }
            .theme-toggle { top: 10px; right: 10px; width: 36px; height: 36px; font-size: 16px; }
        }
    </style>
</head>
<body>

<!-- ===== THEME TOGGLE ===== -->
<button class="theme-toggle" onclick="toggleTheme()" id="themeToggle">🌙</button>

<!-- ===== PARTICLES CANVAS ===== -->
<canvas id="particles-canvas"></canvas>

<!-- ===== APP ===== -->
<div class="container" id="app"></div>

<script>
// ============================================================
// STATE
// ============================================================

let state = {
    modality: null,
    dataset: null,
    index: 0,
    total: 0,
    currentImageName: '',
    currentSlice: 0,
    modalities: [],
    datasets: [],
    imageData: null,
    maskData: null,
    imageLabels: [],
    rotation: 0,
    opacity: 0.5,
    loaded: false,
    classColors: {},
    isLoading: false
};

let canvas = null;
let ctx = null;
let currentImgObj = null;
let currentMaskObj = null;
let loadingCheckInterval = null;
let isDarkTheme = false;

// ============================================================
// THEME
// ============================================================

function toggleTheme() {
    isDarkTheme = !isDarkTheme;
    document.documentElement.setAttribute('data-theme', isDarkTheme ? 'dark' : 'light');
    document.getElementById('themeToggle').textContent = isDarkTheme ? '☀️' : '🌙';
    localStorage.setItem('theme', isDarkTheme ? 'dark' : 'light');
}

// Load saved theme
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'dark') {
    isDarkTheme = true;
    document.documentElement.setAttribute('data-theme', 'dark');
    document.getElementById('themeToggle').textContent = '☀️';
}

// ============================================================
// PARTICLES
// ============================================================

(function initParticles() {
    const canvas = document.getElementById('particles-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles = [];
    const numParticles = 60;
    for (let i = 0; i < numParticles; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            radius: Math.random() * 2.5 + 1,
            dx: (Math.random() - 0.5) * 0.6,
            dy: (Math.random() - 0.5) * 0.6,
            color: `rgba(102, 126, 234, ${Math.random() * 0.3 + 0.1})`
        });
    }

    function animateParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.x += p.dx;
            p.y += p.dy;
            if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.fill();
        });
        requestAnimationFrame(animateParticles);
    }
    animateParticles();

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
})();

// ============================================================
// API HELPERS
// ============================================================

function apiUrl(endpoint, params = {}) {
    const url = new URL(endpoint, window.location.origin);
    Object.keys(params).forEach(key => url.searchParams.append(key, params[key]));
    return url.toString();
}

async function apiGet(endpoint, params = {}) {
    const res = await fetch(apiUrl(endpoint, params));
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

function showLoading(show) {
    let loader = document.getElementById('globalLoader');
    if (show && !loader) {
        loader = document.createElement('div');
        loader.id = 'globalLoader';
        loader.className = 'loading-overlay';
        loader.innerHTML = `
            <div class="loading-content">
                <div class="spinner"></div>
                <div style="font-size:18px; font-weight:600; margin-bottom:8px;">Loading dataset...</div>
            </div>
        `;
        document.body.appendChild(loader);
    } else if (!show && loader) {
        loader.remove();
    }
}

function generateClassColors(classes) {
    const colors = {};
    const numClasses = classes.length;
    classes.forEach((cls, idx) => {
        if (numClasses === 1) {
            colors[cls] = '#1e90ff';
        } else {
            const hue = ((idx / numClasses) + 0.1) % 1.0;
            const rgb = hsvToRgb(hue, 0.8, 0.9);
            colors[cls] = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
        }
    });
    return colors;
}

function hsvToRgb(h, s, v) {
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    let r, g, b;
    switch (i % 6) {
        case 0: [r, g, b] = [v, t, p]; break;
        case 1: [r, g, b] = [q, v, p]; break;
        case 2: [r, g, b] = [p, v, t]; break;
        case 3: [r, g, b] = [p, q, v]; break;
        case 4: [r, g, b] = [t, p, v]; break;
        case 5: [r, g, b] = [v, p, q]; break;
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

// ============================================================
// MAIN FUNCTIONS
// ============================================================

async function loadModalities() {
    const mods = await apiGet('/modalities');
    state.modalities = mods;
    render();
}

async function selectModality(mod) {
    state.modality = mod;
    state.dataset = null;
    state.loaded = false;
    await fetch(apiUrl('/set_modality', { name: mod }));
    const ds = await apiGet('/datasets');
    state.datasets = ds;
    render();
}

async function selectDataset(ds) {
    state.dataset = ds;
    state.loaded = false;
    state.isLoading = true;
    showLoading(true);
    
    try {
        const response = await fetch(apiUrl('/set_dataset', { name: ds }));
        const data = await response.json();
        
        if (data.status === 'loading') {
            startLoadingCheck();
        } else {
            await loadCurrentImage();
            state.loaded = true;
            state.isLoading = false;
            showLoading(false);
            render();
        }
    } catch (e) {
        console.error('Error loading dataset:', e);
        state.isLoading = false;
        showLoading(false);
    }
}

function startLoadingCheck() {
    if (loadingCheckInterval) clearInterval(loadingCheckInterval);
    
    loadingCheckInterval = setInterval(async () => {
        try {
            const status = await apiGet('/loading_status');
            if (!status.is_loading) {
                clearInterval(loadingCheckInterval);
                loadingCheckInterval = null;
                state.isLoading = false;
                state.total = status.total;
                await loadCurrentImage();
                state.loaded = true;
                showLoading(false);
                render();
            }
        } catch (e) {
            console.error('Error checking loading status:', e);
        }
    }, 500);
}

async function loadCurrentImage() {
    try {
        const info = await apiGet('/info');
        if (info.done) {
            state.total = info.total || 0;
            state.imageData = null;
            state.maskData = null;
            return;
        }
        state.index = info.index;
        state.total = info.total;
        state.currentImageName = info.name;
        state.currentSlice = info.slice || 0;
        
        const labels = await apiGet('/image_labels');
        state.imageLabels = labels;
        state.classColors = generateClassColors(labels);
        
        const imgData = await apiGet('/image_data');
        state.imageData = imgData.image;
        state.maskData = imgData.mask;
        state.rotation = 0;
        
        await drawCanvas();
    } catch (e) {
        console.error('Error loading image:', e);
    }
}

async function drawCanvas() {
    canvas = document.getElementById('mainCanvas');
    if (!canvas) return;
    
    ctx = canvas.getContext('2d');
    canvas.width = 512;
    canvas.height = 512;
    
    // Show loading state on canvas
    canvas.className = 'loading';
    
    if (!state.imageData || !state.imageData.length) {
        ctx.clearRect(0, 0, 512, 512);
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, 512, 512);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '24px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No images found', 256, 256);
        canvas.className = 'loaded';
        return;
    }
    
    currentImgObj = new Image();
    currentMaskObj = new Image();
    
    await new Promise((resolve) => {
        let loaded = 0;
        const checkLoaded = () => { loaded++; if (loaded === 2) { canvas.className = 'loaded'; resolve(); } };
        currentImgObj.onload = checkLoaded;
        currentImgObj.onerror = () => { console.error('Failed to load image'); resolve(); };
        currentMaskObj.onload = checkLoaded;
        currentMaskObj.onerror = () => { console.error('Failed to load mask'); resolve(); };
        currentImgObj.src = 'data:image/png;base64,' + state.imageData;
        currentMaskObj.src = 'data:image/png;base64,' + state.maskData;
    });
    
    applyOverlay();
}

function applyOverlay() {
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    
    if (state.rotation !== 0) {
        ctx.translate(w/2, h/2);
        ctx.rotate(state.rotation * Math.PI / 180);
        ctx.translate(-w/2, -h/2);
    }
    
    if (currentImgObj) {
        ctx.drawImage(currentImgObj, 0, 0, w, h);
    }
    
    if (currentMaskObj) {
        const opacity = state.opacity;
        ctx.globalAlpha = opacity;
        ctx.drawImage(currentMaskObj, 0, 0, w, h);
        ctx.globalAlpha = 1.0;
    }
    
    ctx.restore();
}

function updateOpacity(value) {
    state.opacity = parseFloat(value);
    document.getElementById('opacityValue').innerText = Math.round(state.opacity * 100) + '%';
    applyOverlay();
}

function rotateImage() {
    state.rotation = (state.rotation + 90) % 360;
    applyOverlay();
}

async function nextImage() {
    await fetch(apiUrl('/next'));
    await loadCurrentImage();
    render();
}

async function prevImage() {
    await fetch(apiUrl('/back'));
    await loadCurrentImage();
    render();
}

async function resetView() {
    if (loadingCheckInterval) {
        clearInterval(loadingCheckInterval);
        loadingCheckInterval = null;
    }
    await fetch(apiUrl('/reset'));
    state.modality = null;
    state.dataset = null;
    state.loaded = false;
    state.isLoading = false;
    state.imageData = null;
    state.maskData = null;
    render();
}

// ============================================================
// RENDER
// ============================================================

function render() {
    const app = document.getElementById('app');
    if (!app) return;
    
    let html = `
        <div class="card card-gradient">
            <h3 style="margin:0;color:white;">📁 Dataset Browser <span class="speed-badge">⚡ Ultra-Fast Mode</span></h3>
            <div style="color:white; font-size:13px; margin-top:8px;">
                <span>⚡ Loading first ${state.total || 10} images</span>
                <span style="margin-left:20px;">⌨️ Keyboard: <kbd style="background:rgba(255,255,255,0.2); color:white; padding:2px 8px; border-radius:4px;">←</kbd> <kbd style="background:rgba(255,255,255,0.2); color:white; padding:2px 8px; border-radius:4px;">→</kbd> navigate · <kbd style="background:rgba(255,255,255,0.2); color:white; padding:2px 8px; border-radius:4px;">R</kbd> Rotate</span>
                <span style="margin-left:20px;">🎯 Loading only middle slice from 3D volumes</span>
            </div>
        </div>
    `;
    
    html += `<div class="card"><h4 style="margin:0 0 12px 0;">📂 Select Modality</h4><div class="modal-selectors" id="modalityGrid">`;
    html += state.modalities.map(m => 
        `<button class="selector-btn ${state.modality === m ? 'active' : ''}" onclick="selectModality('${m}')">${m}</button>`
    ).join('');
    html += `</div></div>`;
    
    if (state.modality) {
        html += `<div class="card"><h4 style="margin:0 0 12px 0;">🗂️ Select Dataset</h4><div class="modal-selectors" id="datasetGrid">`;
        if (state.datasets.length === 0) {
            html += `<div style="color:var(--text-secondary); padding:10px;">No datasets found in ${state.modality}</div>`;
        } else {
            html += state.datasets.map(d => 
                `<button class="selector-btn ${state.dataset === d ? 'active' : ''}" onclick="selectDataset('${d}')">${d}</button>`
            ).join('');
        }
        html += `</div></div>`;
    }
    
    if (state.isLoading) {
        html += `
            <div class="card">
                <div style="background:#fef3c7; padding:12px 16px; border-radius:8px; color:#92400e;">
                    ⏳ Loading dataset in background (loading only middle slice)...
                </div>
            </div>
        `;
    }
    
    if (state.dataset && state.total > 0 && state.loaded) {
        const hasImage = state.imageData && state.imageData.length > 0;
        
        html += `
            <div class="card">
                <div class="dataset-badge">
                    📊 ${state.dataset} <span style="font-size:12px; background:rgba(255,255,255,0.2); padding:2px 10px; border-radius:12px;">
                        ${state.total} images
                    </span>
                    ${state.currentSlice > 0 ? `<span class="slice-badge">🎯 Slice ${state.currentSlice}</span>` : ''}
                </div>
                <div class="stats">
                    <div class="stat-item"><strong>${state.index+1}</strong> / ${state.total} images</div>
                    ${state.imageLabels.length > 0 ? `<div class="stat-item">🏷️ <strong>${state.imageLabels.length}</strong> classes</div>` : ''}
                </div>
            </div>
        `;
        
        if (hasImage) {
            html += `
                <div class="main-layout">
                    <div class="left-panel">
                        <div class="card">
                            <div class="info-row" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:15px;">
                                <h3 style="margin:0;">🖼️ ${state.currentImageName || 'No image'}</h3>
                                <span style="font-size:14px; color:var(--text-secondary);">${state.index+1}/${state.total}</span>
                            </div>
                            <canvas id="mainCanvas" width="512" height="512"></canvas>
                            <div class="opacity-slider">
                                <span>Overlay opacity:</span>
                                <input type="range" min="0" max="1" step="0.01" value="0.5" oninput="updateOpacity(this.value)">
                                <span id="opacityValue" style="font-weight:600; min-width:40px;">50%</span>
                            </div>
                            <div class="controls">
                                <button onclick="prevImage()" class="btn-nav">◀ Previous</button>
                                <button onclick="nextImage()" class="btn-nav">Next ▶</button>
                                <button onclick="rotateImage()" class="btn-secondary">🔄 Rotate</button>
                                <button onclick="resetView()" class="btn-secondary">🏠 Reset</button>
                            </div>
                            <div style="margin-top:10px;"><strong>🏷️ Classes:</strong> ${state.imageLabels.map(l => `<span class="badge">${l}</span>`).join('') || '<span style="color:var(--text-secondary);">No masks</span>'}</div>
                            ${state.imageLabels.length > 0 ? `
                                <div class="class-legend">
                                    ${state.imageLabels.map(cls => `
                                        <div class="legend-item">
                                            <span class="color-indicator" style="background-color: ${state.classColors[cls] || '#1e90ff'}"></span>
                                            ${cls}
                                        </div>
                                    `).join('')}
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `;
        } else {
            html += `
                <div class="card">
                    <div style="padding:40px; text-align:center; color:var(--text-secondary);">
                        <h3 style="color:var(--text-primary);">📭 No images found</h3>
                        <p>No images could be loaded from this dataset.</p>
                    </div>
                </div>
            `;
        }
    } else if (state.dataset && state.total === 0 && state.loaded) {
        html += `
            <div class="card">
                <div style="padding:40px; text-align:center; color:var(--text-secondary);">
                    <h3 style="color:var(--text-primary);">📭 No images in this dataset</h3>
                    <p>This dataset doesn't contain any images.</p>
                </div>
            </div>
        `;
    }
    
    app.innerHTML = html;
    
    if (state.dataset && state.imageData && state.imageData.length > 0 && state.loaded) {
        setTimeout(drawCanvas, 100);
    }
}

// ============================================================
// KEYBOARD SHORTCUTS
// ============================================================

document.addEventListener('keydown', async (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    
    if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault();
        await nextImage();
    } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        await prevImage();
    } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        rotateImage();
    }
});

// ============================================================
// START
// ============================================================

loadModalities();
</script>
</body>
</html>"""

# ------------------------------------------------
# MAIN
# ------------------------------------------------







def start_visualization(root='datasets', size=512, host='0.0.0.0', 
                        port=5000, debug=False, max_images=10, workers=8):
    """
    Start the dataset browser with the given parameters.
    This function is meant to be called from other scripts.
    """
    # The loaders read these module-level settings; set them from the arguments.
    global ROOT, SIZE, MAX_IMAGES, WORKERS
    ROOT, SIZE, MAX_IMAGES, WORKERS = root, size, max_images, workers

    app = create_app(root=root, size=size, host=host, port=port, debug=debug)

    print(f"OmniMedSeg dataset browser: http://{host}:{port}")
    print(f"  data root:    {os.path.abspath(root)}")
    print(f"  images/set:   {max_images} (3D volumes: middle slice only)")
    print(f"  display size: {size}x{size}, workers: {workers}")
    print("  keys: right/space = next, left = previous, R = rotate")
    print("Press Ctrl+C to stop.")

    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    args = parse_args()
    start_visualization(root=args.root, size=args.size, host=args.host, port=args.port,
                        debug=args.debug, max_images=args.max_images, workers=args.workers)
