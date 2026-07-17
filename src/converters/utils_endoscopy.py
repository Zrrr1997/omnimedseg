import json, base64, io, zlib
from PIL import Image


def load_class_mapping(classes_json_path):
    with open(classes_json_path, "r") as f:
        config = json.load(f)
    return {
        class_def["title"]: idx + 1  # start at 1 to avoid background=0
        for idx, class_def in enumerate(config.get("classes", []))
        if class_def.get("shape") == "bitmap"
    }

def json_mask_to_png(json_filename, class_mapping=None):
    with open(json_filename, "r") as f:
        data = json.load(f)

    width = data["size"]["width"]
    height = data["size"]["height"]
    mask = Image.new("L", (width, height), 0)

    for obj in data.get("objects", []):
        bitmap = obj.get("bitmap")
        class_title = obj.get("classTitle")

        if not bitmap or "data" not in bitmap or "origin" not in bitmap:
            continue
        if class_mapping is None or class_title not in class_mapping:
            continue

        try:
            compressed_data = base64.b64decode(bitmap["data"])
            decompressed_data = zlib.decompress(compressed_data)
            bitmap_img = Image.open(io.BytesIO(decompressed_data)).convert("L")

            # Assign pixel value according to class ID
            class_id = class_mapping[class_title]
            bitmap_arr = bitmap_img.point(lambda p: class_id if p > 0 else 0)

            origin_x, origin_y = bitmap["origin"]
            mask.paste(bitmap_arr, (origin_x, origin_y), bitmap_img)

        except Exception as e:
            print(f"❌ Error in {json_filename}: {e}")
            continue

    return mask