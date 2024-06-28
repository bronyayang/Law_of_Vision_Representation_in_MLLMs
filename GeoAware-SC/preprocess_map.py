import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.utils_correspondence import resize
from model_utils.extractor_dino import ViTExtractor
from model_utils.extractor_sd import load_model, process_features_and_mask

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def process_and_save_features(file_paths, real_size, img_size, layer, facet, model, aug, extractor_vit, flip=False, angle=0):
    for file_path in tqdm(file_paths, desc="Processing images (Flip: {})".format(flip)):
        img1 = Image.open(file_path).convert('RGB')
        if flip:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        # img1 = edge_pad_rotate_and_crop(img1, angle=angle) # Uncomment this line to enable different rotation
        img1_input = resize(img1, real_size, resize=True, to_pil=True)
        img1 = resize(img1, img_size, resize=True, to_pil=True)

        accumulated_features = {}
        for _ in range(NUM_ENSEMBLE): 
            features1 = process_features_and_mask(model, aug, img1_input, mask=False, raw=True)
            del features1['s2']
            for k in features1:
                accumulated_features[k] = accumulated_features.get(k, 0) + features1[k]

        for k in accumulated_features:
            accumulated_features[k] /= NUM_ENSEMBLE

        subdir_name = 'features' if NUM_ENSEMBLE == 1 else f'features_ensemble{NUM_ENSEMBLE}'
        output_subdir = file_path.replace('JPEGImages', subdir_name).rsplit('/', 1)[0]
        os.makedirs(output_subdir, exist_ok=True)
        
        suffix = '_flip' if flip else ''
        output_path = os.path.join(output_subdir, os.path.splitext(os.path.basename(file_path))[0] + f'_sd{suffix}.pt')
        torch.save(accumulated_features, output_path)

        img1_batch = extractor_vit.preprocess_pil(img1)
        img1_desc_dino = extractor_vit.extract_descriptors(img1_batch.cuda(), layer, facet).permute(0, 1, 3, 2).reshape(1, -1, 60, 60)
        output_path_dino = os.path.join(output_subdir, os.path.splitext(os.path.basename(file_path))[0] + f'_dino{suffix}.pt')
        torch.save(img1_desc_dino, output_path_dino)

if __name__ == '__main__':
    # Configuration
    set_seed()
    base_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/SPair-71k/JPEGImages'
    real_size, img_size, layer, facet = 960, 840, 11, 'token'
    NUM_ENSEMBLE = 1

    # Load models
    model, aug = load_model(diffusion_ver='v1-5', image_size=real_size, num_timesteps=50, block_indices=[2,5,8,11])
    extractor_vit = ViTExtractor('dinov2_vitb14', 14, device='cuda')

    all_files = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(base_dir) for file in files if file.endswith('.jpg')]

    angles = [0] # angles for rotation
    for angle in angles:
        # Process and save features
        process_and_save_features(all_files, real_size, img_size, layer, facet, model, aug, extractor_vit, flip=False, angle=angle)
        process_and_save_features(all_files, real_size, img_size, layer, facet, model, aug, extractor_vit, flip=True, angle=angle)

    print("All processing done.")