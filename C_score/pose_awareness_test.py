import torch
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from model_utils.extractor_dino import ViTExtractor
from model_utils.extractor_sd import load_model, process_features_and_mask, get_mask
from utils.utils_correspondence import co_pca, resize
from utils.logger import get_logger
from loguru import logger

# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument("--FUSE_DINO", type=int, default=1,
                    help="Specify the value for FUSE_DINO")
parser.add_argument("--ONLY_DINO", type=int, default=0,
                    help="Specify the value for ONLY_DINO")
# Parse the arguments
args = parser.parse_args()

VER = "v1-5"
PCA = False
CO_PCA = True
PCA_DIMS = [256, 256, 256]
SIZE =960
RESOLUTION = 256
EDGE_PAD = False
# Now you can use the values specified at the command line
FUSE_DINO = args.FUSE_DINO
ONLY_DINO = args.ONLY_DINO
DINOV2 = True
MODEL_SIZE = 'base' # 'small' or 'base', indicate dinov2 model
TEXT_INPUT = False
SEED = 42
TIMESTEP = 100 #flexible from 0~200
RESOLUTION = 128
DIST = 'l2' if FUSE_DINO and not ONLY_DINO else 'cos'
if ONLY_DINO:
    FUSE_DINO = True

logger = get_logger(f'results_pose-awareness/result_SD_{not ONLY_DINO}_DINO_{bool(FUSE_DINO)}.log')
logger.info(args)

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

model, aug = load_model(diffusion_ver=VER, image_size=SIZE, num_timesteps=TIMESTEP, decoder_only=False)
stride = 14 if DINOV2 else 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_dict={'small':'dinov2_vits14',
                'base':'dinov2_vitb14',
                'large':'dinov2_vitl14',
                'giant':'dinov2_vitg14'}
model_type = model_dict[MODEL_SIZE] if DINOV2 else 'dino_vits8'
vit_extractor = ViTExtractor(model_type, stride, device=device)

def compute_pair_feature(model, aug, category, dist='cos', real_size=960):
    if type(category) == str:
        category = [category]
    img_size = 840 if DINOV2 else 244
    layer = 11 if DINOV2 else 9
    if 'l' in model_type:
        layer = 23
    elif 'g' in model_type:
        layer = 39
    facet = 'token' if DINOV2 else 'key'

    patch_size = vit_extractor.model.patch_embed.patch_size[0] if DINOV2 else vit_extractor.model.patch_embed.patch_size
    num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)
    
    input_text = "a photo of "+category[-1][0] if TEXT_INPUT else None

    result = []

    # Load image 1
    img1 = Image.open(src_path).convert('RGB')
    img1_input = resize(img1, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
    img1 = resize(img1, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

    # Load image 2
    img2 = Image.open(trg_path).convert('RGB')
    img2_input = resize(img2, real_size, resize=True, to_pil=True, edge=EDGE_PAD)
    img2 = resize(img2, img_size, resize=True, to_pil=True, edge=EDGE_PAD)

    with torch.no_grad():
        if not ONLY_DINO:
            if src_path in src_feat_map_cache.keys():
                features1 = src_feat_map_cache[src_path]['sd']
            else:
                features1 = process_features_and_mask(model, aug, img1_input, input_text=input_text,  mask=False, raw=True)
            if len(trg_feat_map_cache.keys())>0:
                features2 = trg_feat_map_cache['sd']
            else:
                features2 = process_features_and_mask(model, aug, img2_input, input_text=input_text,  mask=False, raw=True)
            processed_features1, processed_features2 = co_pca(features1, features2, PCA_DIMS)
            img1_desc = processed_features1.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
            img2_desc = processed_features2.reshape(1, 1, -1, num_patches**2).permute(0,1,3,2)
        if FUSE_DINO:
            if src_path in src_feat_map_cache.keys():
                img1_desc_dino = src_feat_map_cache[src_path]['dino']
            else:
                img1_batch = vit_extractor.preprocess_pil(img1)
                img1_desc_dino = vit_extractor.extract_descriptors(img1_batch.to(device), layer, facet)
            if len(trg_feat_map_cache.keys())>0:
                img2_desc_dino = trg_feat_map_cache['dino']
            else:
                img2_batch = vit_extractor.preprocess_pil(img2)
                img2_desc_dino = vit_extractor.extract_descriptors(img2_batch.to(device), layer, facet)
            
        if dist == 'l1' or dist == 'l2':
            # normalize the features
            img1_desc = img1_desc / img1_desc.norm(dim=-1, keepdim=True)
            img2_desc = img2_desc / img2_desc.norm(dim=-1, keepdim=True)
            if FUSE_DINO:
                img1_desc_dino = img1_desc_dino / img1_desc_dino.norm(dim=-1, keepdim=True)
                img2_desc_dino = img2_desc_dino / img2_desc_dino.norm(dim=-1, keepdim=True)

        if FUSE_DINO and not ONLY_DINO:
            # cat two features together
            img1_desc = torch.cat((img1_desc, img1_desc_dino), dim=-1)
            img2_desc = torch.cat((img2_desc, img2_desc_dino), dim=-1)

        if ONLY_DINO:
            img1_desc = img1_desc_dino
            img2_desc = img2_desc_dino

        if src_path in src_feat_map_cache.keys():
            mask1 = src_feat_map_cache[src_path]['mask']
        else:
            mask1 = get_mask(model, aug, img1, category[0])
            # update the src_feat_map_cache
            if not ONLY_DINO and FUSE_DINO:
                src_feat_map_cache[src_path] = {'sd':features1, 'dino':img1_desc_dino, 'mask':mask1}
            elif ONLY_DINO:
                src_feat_map_cache[src_path] = {'dino':img1_desc_dino, 'mask':mask1}
            elif not FUSE_DINO:
                src_feat_map_cache[src_path] = {'sd':features1, 'mask':mask1}

        if len(trg_feat_map_cache.keys())>0:
            mask2 = trg_feat_map_cache['mask']
        else:
            mask2 = get_mask(model, aug, img2, category[-1])
            # update the trg_feat_map_cache
            if not ONLY_DINO and FUSE_DINO:
                trg_feat_map_cache['sd'] = features2
                trg_feat_map_cache['dino'] = img2_desc_dino
                trg_feat_map_cache['mask'] = mask2
            elif ONLY_DINO:
                trg_feat_map_cache['dino'] = img2_desc_dino
                trg_feat_map_cache['mask'] = mask2
            elif not FUSE_DINO:
                trg_feat_map_cache['sd'] = features2
                trg_feat_map_cache['mask'] = mask2

        result.append([img1_desc.cpu(), img2_desc.cpu(), mask1.cpu(), mask2.cpu()])

    return result

def process_images(src_path, trg_path, categories):

    result = compute_pair_feature(model, aug, category=categories, dist=DIST)
    
    # high resolution swap, will take the instance of interest from the target image and replace it in the source image
    feature2,feature1,mask2,mask1 = result[0]
    src_feature_reshaped = feature1.squeeze().permute(1,0).reshape(1,-1,60,60).cuda()
    tgt_feature_reshaped = feature2.squeeze().permute(1,0).reshape(1,-1,60,60).cuda()
    src_img=Image.open(trg_path).convert('RGB')
    tgt_img=Image.open(src_path).convert('RGB')
    
    patch_size = RESOLUTION # the resolution of the output image, set to 256 could be faster
    src_img = resize(src_img, patch_size, resize=True, to_pil=False, edge=EDGE_PAD)
    tgt_img = resize(tgt_img, patch_size, resize=True, to_pil=False, edge=EDGE_PAD)
    resized_src_mask = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size), mode='nearest').squeeze().cuda()
    resized_tgt_mask = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size), mode='nearest').squeeze().cuda()
    src_feature_upsampled = F.interpolate(src_feature_reshaped, size=(patch_size, patch_size), mode='bilinear').squeeze()
    tgt_feature_upsampled = F.interpolate(tgt_feature_reshaped, size=(patch_size, patch_size), mode='bilinear').squeeze()
    # mask the feature
    src_feature_upsampled = src_feature_upsampled * resized_src_mask.repeat(src_feature_upsampled.shape[0],1,1)
    # tgt_feature_upsampled = tgt_feature_upsampled * resized_tgt_mask.repeat(src_feature_upsampled.shape[0],1,1)
    # Set the masked area to a very small number
    src_feature_upsampled[src_feature_upsampled == 0] = -100000
    # tgt_feature_upsampled[tgt_feature_upsampled == 0] = -100000
    # Calculate the cosine similarity between src_feature and tgt_feature
    src_features_2d=src_feature_upsampled.reshape(src_feature_upsampled.shape[0],-1).permute(1,0)
    tgt_features_2d=tgt_feature_upsampled.reshape(tgt_feature_upsampled.shape[0],-1).permute(1,0)
    # swapped_image=src_img
    min_dist = []
    for patch_idx in range(patch_size*patch_size):
        # If the patch is in the resized_src_mask_out_layers, find the corresponding patch in the target_output and swap them
        if resized_src_mask[patch_idx // patch_size, patch_idx % patch_size] == 1:
            # Find the corresponding patch with the highest cosine similarity
            distances = torch.linalg.norm(tgt_features_2d - src_features_2d[patch_idx], dim=1)
            tgt_patch_idx = torch.argmin(distances)
            min_dist.append(distances[tgt_patch_idx].item())

    min_dist = torch.tensor(min_dist)
    min_dist = min_dist.mean()

    return result, min_dist

img_to_pose={
    "data/SPair-71k/JPEGImages/cat/2007_001825.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2007_002597.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2007_005460.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2007_006303.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2007_009221.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2008_000112.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2008_000115.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2008_000227.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2008_000345.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2008_000464.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2008_001078.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2008_001640.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2008_001885.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2008_002215.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2008_002597.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2008_002682.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2008_003499.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2008_003519.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2008_005252.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2008_005380.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2008_005386.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2008_005421.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2008_005699.jpg":"b",
    "data/SPair-71k/JPEGImages/cat/2008_006175.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2008_006190.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2008_006325.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2008_006728.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2008_006973.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2008_006999.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2008_007403.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2008_007726.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2009_000599.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2009_000716.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2009_002228.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2009_002592.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2009_002813.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2009_003056.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2009_003129.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2009_003771.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2009_004051.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2009_004234.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2009_004291.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2009_004382.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2009_004887.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2009_004940.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2009_004983.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2009_005037.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2009_005095.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_000009.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2010_000054.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_000099.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_000109.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_000163.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_000218.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_000244.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_000291.jpg":"b",
    "data/SPair-71k/JPEGImages/cat/2010_000374.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_000439.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_000468.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2010_000469.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_000576.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_000616.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_000702.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_000872.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_001382.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_001386.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_001468.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_001590.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_001647.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_001994.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2010_002026.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_002098.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_002224.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_002504.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_002531.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_002909.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2010_002993.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_003174.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_003402.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_003483.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_003488.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2010_003539.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_004717.jpg":"b",
    "data/SPair-71k/JPEGImages/cat/2010_004768.jpg":"b",
    "data/SPair-71k/JPEGImages/cat/2010_004829.jpg":"b",
    "data/SPair-71k/JPEGImages/cat/2010_004933.jpg":"b",
    "data/SPair-71k/JPEGImages/cat/2010_004954.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_005115.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_005216.jpg":"b",
    "data/SPair-71k/JPEGImages/cat/2010_005394.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2010_005408.jpg":"r",
    "data/SPair-71k/JPEGImages/cat/2010_005763.jpg":"b",
    "data/SPair-71k/JPEGImages/cat/2010_005853.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2010_006040.jpg":"b",
    "data/SPair-71k/JPEGImages/cat/2011_000426.jpg":"b",
    "data/SPair-71k/JPEGImages/cat/2011_000469.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2011_000973.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2011_000999.jpg":"f",
    "data/SPair-71k/JPEGImages/cat/2011_001568.jpg":"l",
    "data/SPair-71k/JPEGImages/cat/2011_002034.jpg":"r"
    }

src_paths = sorted(glob.glob("data/images/pose_template/*.png"))
src_feat_map_cache = {}
trg_paths = sorted(glob.glob("data/SPair-71k/JPEGImages/cat/*.jpg"))

# Constants
pose_indices = {
    'back': [0, 4, 8],
    'front': [1, 5, 9],
    'left': [2, 6, 10],
    'right': [3, 7, 11]
}

# Initialize counters
correct_counts = {'2': 0, '4': 0, '2_lr': 0, '2_fb': 0, '4_lr': 0, '4_fb': 0}
view_counts = {'lr': 0, 'bf': 0}

for trg_path in tqdm(trg_paths):
    distances = []
    trg_feat_map_cache = {}
    for src_path in src_paths:
        _, dis = process_images(src_path, trg_path, categories=[['cat'], ['cat']])
        distances.append(dis)

    distances = torch.tensor(distances)
    pose_distances = {key: distances[indices] for key, indices in pose_indices.items()}
    bf_predict = ((pose_distances['back'] - pose_distances['front'])>0).sum() > 1
    lr_predict = ((pose_distances['left'] - pose_distances['right'])>0).sum() > 1
    
    bf_indicator = 'f' if bf_predict else 'b'
    lr_indicator = 'r' if lr_predict else 'l'
    prediction_2 = bf_indicator if bf_indicator in img_to_pose[trg_path] else lr_indicator
    prediction_4 = torch.argmin(distances.reshape(3, 4), dim=-1).bincount().argmax().item()
    bflr_indicator = ['b', 'f', 'l', 'r', 'x'][prediction_4]

    # Update correct counts
    correct_counts['2'] += img_to_pose[trg_path] in [bf_indicator, lr_indicator]
    if img_to_pose[trg_path] not in [bf_indicator, lr_indicator]:
        logger.info(f'{trg_path} is not correctly predicted for 2 views, gt: {img_to_pose[trg_path]}, pred: {bf_indicator + lr_indicator}')
    correct_counts['4'] += img_to_pose[trg_path] == bflr_indicator

    pose = img_to_pose[trg_path]
    if pose in ['b', 'f']:
        view_counts['bf'] += 1
        correct_counts['2_fb'] += bf_indicator == pose
        correct_counts['4_fb'] += bflr_indicator == pose
    elif pose in ['l', 'r']:
        view_counts['lr'] += 1
        correct_counts['2_lr'] += lr_indicator == pose
        correct_counts['4_lr'] += bflr_indicator == pose

# Log results
total = len(trg_paths)
logger.info(f"correct_2: {correct_counts['2']/total}, correct_4: {correct_counts['4']/total}")
logger.info(f"correct_2_lr: {correct_counts['2_lr']/view_counts['lr']}, correct_2_fb: {correct_counts['2_fb']/view_counts['bf']}")
logger.info(f"correct_4_lr: {correct_counts['4_lr']/view_counts['lr']}, correct_4_fb: {correct_counts['4_fb']/view_counts['bf']}")