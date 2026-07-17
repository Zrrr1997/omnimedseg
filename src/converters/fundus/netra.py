import os 
import cv2
from tqdm import tqdm
from src.utils.dataset_utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_file_netra(args):
    idx, file, converted_path, ambiguous_cases = args

    for ac in ambiguous_cases:
        if ac in file:
            return None  # skip ambiguous

    filename = f"{idx:06d}.png"

    img = cv2.imread(file)
    if img is None:
        return None

    if len(img.shape) == 3 and np.allclose(img[:, :, 0], img[:, :, 1]) and np.allclose(img[:, :, 1], img[:, :, 2]):
        return None

    label_path = file.replace('NETRA-RAW', 'NETRA-ANNOTATED')
    label = cv2.imread(label_path)
    if label is None:
        print(f"⚠️ Could not read label for {file}")
        return None

    if img.shape != label.shape:
        img = register_img_label(img, label)
    assert label.shape == img.shape

    diff = cv2.absdiff(img, label)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    label = (gray > np.percentile(gray, 99)).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(label, connectivity=8)

    min_area = 100
    cleaned = np.zeros_like(label)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    label = cleaned

    if np.sum(label) == 0 or '442' in file:
        return None

    outer, inner = prepare_label(label, file)

    outer = (outer > 0).astype(np.uint8) * 255
    inner = (inner > 0).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(converted_path, 'labels', 'optic_disc', filename), outer)
    cv2.imwrite(os.path.join(converted_path, 'labels', 'optic_cup', filename), inner)
    cv2.imwrite(os.path.join(converted_path, 'images', filename), img)

    return filename, {
        'image': file,
        'optic_disc': label_path,
        'optic_cup': label_path
    }

def convert_netra():
    converted_path = os.path.join('datasets', 'Fundus', 'NETRA', 'converted')
    ensure_directories(converted_path, ['optic_disc', 'optic_cup'])

    extracted_base = os.path.join('datasets', 'Fundus', 'NETRA', 'extracted')
    raw_path = os.path.join(extracted_base, 'NETRA-RAW')

    ambiguous_cases = ['212', '292', '347', '350', '441', '507', '521', '551', '556', '567', '577']

    image_files = sorted([os.path.join(raw_path, f) for f in os.listdir(raw_path)])

    args_list = [(idx, file, converted_path, ambiguous_cases) for idx, file in enumerate(image_files)]

    filename_map = {}

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file_netra, args_list), total=len(args_list), desc="🖼️  Processing NETRA images"))

    for result in results:
        if result is not None:
            filename, data = result
            filename_map[filename] = data

    save_mapping(converted_path, filename_map)

def prepare_label(label, fn):
    edge_cases = ['274', '399', '555', '568', '577']

    edge_case = False
    for ec in edge_cases:
        if ec in fn:
            edge_case = True

    import cc3d
    # Ensure it's binary
    label = (label > 0).astype(np.uint8)

    # 1. Set 15-pixel border to 0
    label[:15, :] = 0
    label[-15:, :] = 0
    label[:, :15] = 0
    label[:, -15:] = 0

    # 2. Connected components
    components = cc3d.connected_components(label, connectivity=8)
    num_components = components.max()

    if num_components >= 2 and not edge_case:
        # Count sizes and find the two largest
        sizes = [(i, np.sum(components == i)) for i in range(1, num_components + 1)]
        sizes = sorted(sizes, key=lambda x: x[1], reverse=True)[:2]
        mask1 = (components == sizes[0][0]).astype(np.uint8) * 255
        mask2 = (components == sizes[1][0]).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(contours[0])
        hull_mask_outer = np.zeros_like(label, dtype=np.uint8)
        cv2.drawContours(hull_mask_outer, [hull], -1, 255, thickness=-1)

        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(contours[0])
        hull_mask_inner = np.zeros_like(label, dtype=np.uint8)
        cv2.drawContours(hull_mask_inner, [hull], -1, 255, thickness=-1)
        return hull_mask_outer, hull_mask_inner

    elif num_components == 1 or edge_case:
        if edge_case: # additional scribble that needs to be removed first
            sizes = [(i, np.sum(components == i)) for i in range(1, num_components + 1)]
            sizes = sorted(sizes, key=lambda x: x[1], reverse=True)[:2]
            label = (components == sizes[0][0]).astype(np.uint8) * 255
        # 3. Convex hull if just one component
        contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        hull = cv2.convexHull(contours[0])
        hull_mask_outer = np.zeros_like(label, dtype=np.uint8)
        cv2.drawContours(hull_mask_outer, [hull], -1, 255, thickness=-1)


        hull_edges = np.zeros_like(label, dtype=np.uint8)
        cv2.drawContours(hull_edges, [hull], -1, 255, thickness=1)

        # Dilate hull edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37))
        dilated_edges = cv2.dilate(hull_edges, kernel)

        # Remove hull edges from label
        cleaned = cv2.bitwise_and(label, cv2.bitwise_not(dilated_edges))

        # Keep largest remaining component
        components = cc3d.connected_components(cleaned, connectivity=8)
        sizes = [(i, np.sum(components == i)) for i in range(1, components.max() + 1)]
        if not sizes:
            return None, None
        largest_id = max(sizes, key=lambda x: x[1])[0]
        result = (components == largest_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(contours[0])
        hull_mask_inner = np.zeros_like(label, dtype=np.uint8)
        cv2.drawContours(hull_mask_inner, [hull], -1, 255, thickness=-1)

        return hull_mask_outer, hull_mask_inner

    return None, None


def register_img_label(img, label):
    """
    Finds the position of label in the image and returns the corresponding crop of the image.
    Assumes label is a crop from the image.
    """
    # Convert to grayscale for matching
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label_gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv2.matchTemplate(img_gray, label_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Get the top-left corner of the best match
    top_left = max_loc
    h, w = label.shape[:2]

    # Crop the corresponding region from the original image
    registered_crop = img[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]

    return registered_crop