import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
import json
# Ensure the segment_anything package is accessible.
sys.path.append("..")
"""pip install git+https://github.com/facebookresearch/segment-anything.git"""
from segment_anything import sam_model_registry, SamPredictor
from utils.utils_correspondence import resize

# Determine base directory from command line argument or default.
try:
    base_dir = sys.argv[1]
except IndexError:
    base_dir = 'data/ap-10k/JPEGImages'
SPAIR = "SPair-71k" in base_dir

"""wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"""
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
predictor = SamPredictor(sam)
ANNO_SIZE = 960

def preprocess_kps_pad(kps, img_width, img_height, size):
    """Adjust keypoints for images after padding to maintain their accuracy."""
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    offset_x, offset_y = 0, 0
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = (size - new_h) // 2
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = (size - new_w) // 2
    kps[:, 0] += offset_x
    kps[:, 1] += offset_y
    return kps

# Collect all image file paths.
all_files = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(base_dir) for file in files if file.endswith('.jpg')]
all_files.sort()

for input_path in tqdm(all_files, desc="Processing images"):
    pil_image = Image.open(input_path).convert('RGB')
    width, height = pil_image.size
    pil_image = resize(pil_image, ANNO_SIZE)  # Resizing for uniformity.
    image = np.array(pil_image)  # Convert to OpenCV format to work with SAM model.

    # Load JSON data.
    json_path = input_path.replace('jpg', 'json').replace("JPEGImages", "ImageAnnotation")
    with open(json_path) as f:
        data = json.load(f)
        input_box = np.array(data["bbox"] if not SPAIR else data["bndbox"])
        if not SPAIR:
            input_box[-2:] += input_box[:2]

    # Adjust bounding box for resized image.
    input_box = preprocess_kps_pad(torch.tensor(input_box).reshape(2,2).float(), width, height, ANNO_SIZE).reshape(-1).numpy()
    predictor.set_image(image)
    masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)

    # Prepare output paths for saving masks.
    output_subdir = os.path.dirname(input_path).replace('JPEGImages', 'features')
    os.makedirs(output_subdir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_subdir, f'{base_filename}_mask.png')
    output_path_flip = os.path.join(output_subdir, f'{base_filename}_mask_flip.png')

    # Save masks and flipped masks.
    cv2.imwrite(output_path, np.clip(masks[0], 0, 1) * 255)
    flip_mask = np.flip(masks[0], axis=1)  # Flip the mask horizontally.
    cv2.imwrite(output_path_flip, flip_mask * 255)