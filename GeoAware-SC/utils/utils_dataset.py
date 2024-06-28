import json
import os
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
from glob import glob
from PIL import Image
from torch.nn.functional import pad as F_pad
from tqdm import tqdm
# General

def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3].clone()  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale

def load_and_prepare_data(args):
    """
    Load and prepare dataset for training.

    Parameters:
    - PASCAL_TRAIN: Flag to indicate if training on PASCAL dataset.
    - AP10K_TRAIN: Flag to indicate if training on AP10K dataset.
    - BBOX_THRE: Flag to indicate if bounding box thresholds are used.
    - ANNO_SIZE: Annotation size.
    - SAMPLE: Sampling rate for the dataset.

    Returns:
    - files: List of file paths.
    - kps: Keypoints tensor.
    - cats: Categories tensor.
    - used_points_set: Used points set.
    - all_thresholds (optional): All thresholds.
    """

    # Determining the data directory and categories based on the training dataset
    if args.TRAIN_DATASET=='pascal':
        data_dir = 'data/PF-dataset-PASCAL'
        categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                      'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif args.TRAIN_DATASET=='ap10k':
        data_dir = 'data/ap-10k'
        subfolders = os.listdir(os.path.join(data_dir, 'ImageAnnotation'))
        categories = sorted([item for subfolder in subfolders for item in os.listdir(os.path.join(data_dir, 'ImageAnnotation', subfolder))])
    else:
        data_dir = 'data/SPair-71k'
        categories = sorted(os.listdir(os.path.join(data_dir, 'ImageAnnotation')))

    files, kps, cats, used_points_set, all_thresholds = ([] for _ in range(5))

    # Loading data based on the dataset and preprocessing it
    for cat_idx, cat in tqdm(enumerate(categories), total=len(categories), desc="Processing Categories"):
        if args.TRAIN_DATASET=='pascal':
            single_files, single_kps, thresholds, used_points = load_pascal_data(data_dir, size=args.ANNO_SIZE, category=cat, split='train', subsample=args.SAMPLE)
        elif args.TRAIN_DATASET=='ap10k':
            if cat in ['argali sheep', 'black bear', 'king cheetah']:
                continue
            single_files, single_kps, thresholds, used_points = load_ap10k_data(data_dir, size=args.ANNO_SIZE, category=cat, split='trn', subsample=args.SAMPLE)
        else:
            single_files, single_kps, thresholds, used_points = load_spair_data(data_dir, size=args.ANNO_SIZE, category=cat, split='trn', subsample=args.SAMPLE)
        
        files.extend(single_files)
        single_kps = F_pad(single_kps, (0, 0, 0, 30 - single_kps.shape[1], 0, 0), value=0)
        kps.append(single_kps)
        used_points_set.extend([used_points] * (len(single_files) // 2))
        cats.extend([cat_idx] * (len(single_files) // 2))
        if args.BBOX_THRE:
            all_thresholds.extend(thresholds)
    kps = torch.cat(kps, dim=0)

    # Shuffling the data
    shuffled_files, shuffled_kps, shuffled_cats, shuffled_used_points_set, shuffled_thresholds = shuffle_data(files, kps, cats, used_points_set, all_thresholds, args.BBOX_THRE)

    return shuffled_files, shuffled_kps, shuffled_cats, shuffled_used_points_set, shuffled_thresholds if args.BBOX_THRE else None

def shuffle_data(files, kps, cats, used_points_set, all_thresholds, BBOX_THRE):
    """
    Shuffle dataset pairs.

    Parameters are lists of files, keypoints, categories, used points, all thresholds, and a flag for bounding box thresholds.
    Returns shuffled lists.
    """
    pair_count = len(files) // 2
    pair_indices = torch.randperm(pair_count)
    actual_indices = pair_indices * 2

    shuffled_files = [files[idx] for i in actual_indices for idx in [i, i+1]]
    shuffled_kps = torch.cat([kps[idx:idx+2] for idx in actual_indices])
    shuffled_cats = [cats[i//2] for i in actual_indices]
    shuffled_used_points_set = [used_points_set[i//2] for i in actual_indices]
    shuffled_thresholds = [all_thresholds[idx] for i in actual_indices for idx in [i, i+1]] if BBOX_THRE else []

    return shuffled_files, shuffled_kps, shuffled_cats, shuffled_used_points_set, shuffled_thresholds

def load_eval_data(args, path, category, split):
    if args.EVAL_DATASET == 'ap10k':
        files, kps, thresholds, used_kps = load_ap10k_data(path, args.ANNO_SIZE, category, split, args.TEST_SAMPLE)
    elif args.EVAL_DATASET == 'pascal':
        files, kps, thresholds, used_kps = load_pascal_data(path, args.ANNO_SIZE, category, split, args.TEST_SAMPLE)
    else:
        files, kps, thresholds, used_kps = load_spair_data(path, args.ANNO_SIZE, category, split, args.TEST_SAMPLE)

    return files, kps, thresholds, used_kps

def get_dataset_info(args, split):
    if args.EVAL_DATASET == 'pascal':
        data_dir = 'data/PF-dataset-PASCAL'
        categories = sorted(os.listdir(os.path.join(data_dir, 'Annotations')))
    elif args.EVAL_DATASET == 'ap10k':
        data_dir = 'data/ap-10k'
        categories = []
        subfolders = os.listdir(os.path.join(data_dir, 'ImageAnnotation'))
        # Handle AP10K_EVAL test settings
        if args.AP10K_EVAL_SUBSET == 'intra-species':
            categories = [folder for subfolder in subfolders for folder in os.listdir(os.path.join(data_dir, 'ImageAnnotation', subfolder))]
        elif args.AP10K_EVAL_SUBSET == 'cross-species':
            categories = [subfolder for subfolder in subfolders if len(os.listdir(os.path.join(data_dir, 'ImageAnnotation', subfolder))) > 1]
            split += '_cross_species'
        elif args.AP10K_EVAL_SUBSET == 'cross-family':
            categories = ['all']
            split += '_cross_family'
        categories = sorted(categories)
    else: # SPair
        data_dir = 'data/SPair-71k'
        categories = sorted(os.listdir(os.path.join(data_dir, 'ImageAnnotation')))

    return data_dir, categories, split

# AP-10K

def load_ap10k_data(path="data/ap-10k", size=840, category='cat', split='test', subsample=20):
    np.random.seed(42)
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*:{category}.json'))
    if subsample is not None and subsample > 0:
        pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
    files = []
    kps = []
    thresholds = []
    for pair in pairs:
        with open(pair) as f:
            data = json.load(f)
        source_json_path = data["src_json_path"]
        target_json_path = data["trg_json_path"]
        src_img_path = source_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')
        trg_img_path = target_json_path.replace("json", "jpg").replace('ImageAnnotation', 'JPEGImages')

        with open(source_json_path) as f:
            src_file = json.load(f)
        with open(target_json_path) as f:
            trg_file = json.load(f)
            
        source_bbox = np.asarray(src_file["bbox"])  # l t w h
        target_bbox = np.asarray(trg_file["bbox"])
        
        source_size = np.array([src_file["width"], src_file["height"]])  # (W, H)
        target_size = np.array([trg_file["width"], trg_file["height"]])  # (W, H)

        # print(source_raw_kps.shape)
        source_kps = torch.tensor(src_file["keypoints"]).view(-1, 3).float()
        source_kps[:,-1] /= 2
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)

        target_kps = torch.tensor(trg_file["keypoints"]).view(-1, 3).float()
        target_kps[:,-1] /= 2
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)
        # The source thresholds aren't actually used to evaluate PCK on SPair-71K, but for completeness
        # they are computed as well:
        # thresholds.append(max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0]))
        if 'test' in split:
            thresholds.append(max(target_bbox[3], target_bbox[2])*trg_scale)
        elif 'trn' in split:
            thresholds.append(max(source_bbox[3], source_bbox[2])*src_scale)
            thresholds.append(max(target_bbox[3], target_bbox[2])*trg_scale)

        kps.append(source_kps)
        kps.append(target_kps)
        files.append(src_img_path)
        files.append(trg_img_path)

    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    # print(f'Final number of used key points: {kps.size(1)}')
    return files, kps, thresholds, used_kps

# SPair-71K

def load_spair_data(path="data/SPair-71k", size=256, category='cat', split='test', subsample=None):
    np.random.seed(42)
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*:{category}.json'))
    if subsample is not None and subsample > 0:
        pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
    files = []
    thresholds = []
    kps = []
    category_anno = list(glob(f'{path}/ImageAnnotation/{category}/*.json'))[0]
    with open(category_anno) as f:
        num_kps = len(json.load(f)['kps'])
    for pair in pairs:
        source_kps = torch.zeros(num_kps, 3)
        target_kps = torch.zeros(num_kps, 3)
        with open(pair) as f:
            data = json.load(f)
        assert category == data["category"]
        source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
        target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'
        source_json_name = source_fn.replace('JPEGImages','ImageAnnotation').replace('jpg','json')
        target_json_name = target_fn.replace('JPEGImages','ImageAnnotation').replace('jpg','json')
        source_bbox = np.asarray(data["src_bndbox"])    # (x1, y1, x2, y2)
        target_bbox = np.asarray(data["trg_bndbox"])
        with open(source_json_name) as f:
            file = json.load(f)
            kpts_src = file['kps']
        with open(target_json_name) as f:
            file = json.load(f)
            kpts_trg = file['kps']

        source_size = data["src_imsize"][:2]  # (W, H)
        target_size = data["trg_imsize"][:2]  # (W, H)

        for i in range(30):
            point = kpts_src[str(i)]
            if point is None:
                source_kps[i, :3] = 0
            else:
                source_kps[i, :2] = torch.Tensor(point).float()  # set x and y
                source_kps[i, 2] = 1
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)
        
        for i in range(30):
            point = kpts_trg[str(i)]
            if point is None:
                target_kps[i, :3] = 0
            else:
                target_kps[i, :2] = torch.Tensor(point).float()
                target_kps[i, 2] = 1
        # target_raw_kps = torch.cat([torch.tensor(data["trg_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        # target_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=target_raw_kps)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)
        if split == 'test' or split == 'val':
            thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0])*trg_scale)
        elif split == 'trn':
            thresholds.append(max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0])*src_scale)
            thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0])*trg_scale)

        kps.append(source_kps)
        kps.append(target_kps)
        files.append(source_fn)
        files.append(target_fn)
    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]

    return files, kps, thresholds, used_kps

# Pascal

def read_mat(path, obj_name):
    r"""Reads specified objects from Matlab data file, (.mat)"""
    mat_contents = sio.loadmat(path)
    mat_obj = mat_contents[obj_name]

    return mat_obj

def process_kps_pascal(kps):
    # Step 1: Reshape the array to (20, 2) by adding nan values
    num_pad_rows = 20 - kps.shape[0]
    if num_pad_rows > 0:
        pad_values = np.full((num_pad_rows, 2), np.nan)
        kps = np.vstack((kps, pad_values))
        
    # Step 2: Reshape the array to (20, 3) 
    # Add an extra column: set to 1 if the row does not contain nan, 0 otherwise
    last_col = np.isnan(kps).any(axis=1)
    last_col = np.where(last_col, 0, 1)
    kps = np.column_stack((kps, last_col))

    # Step 3: Replace rows with nan values to all 0's
    mask = np.isnan(kps).any(axis=1)
    kps[mask] = 0

    return torch.tensor(kps).float()

def load_pascal_data(path="data/PF-dataset-PASCAL", size=256, category='cat', split='test', subsample=None):
    
    def get_points(point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=";")
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=";")
        Xpad = -np.ones(20)
        Xpad[: len(X)] = X
        Ypad = -np.ones(20)
        Ypad[: len(X)] = Y
        Zmask = np.zeros(20)
        Zmask[: len(X)] = 1
        point_coords = np.concatenate(
            (Xpad.reshape(1, 20), Ypad.reshape(1, 20), Zmask.reshape(1,20)), axis=0
        )
        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords
    
    np.random.seed(42)
    files = []
    kps = []
    test_data = pd.read_csv(f'{path}/{split}_pairs_pf_pascal.csv')
    cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    cls_ids = test_data.iloc[:,2].values.astype("int") - 1
    cat_id = cls.index(category)
    subset_id = np.where(cls_ids == cat_id)[0]
    # logger.info(f'Number of Pairs for {category} = {len(subset_id)}')
    subset_pairs = test_data.iloc[subset_id,:]
    src_img_names = np.array(subset_pairs.iloc[:,0])
    trg_img_names = np.array(subset_pairs.iloc[:,1])
    # print(src_img_names.shape, trg_img_names.shape)
    if not split.startswith('train'):
        point_A_coords = subset_pairs.iloc[:,3:5]
        point_B_coords = subset_pairs.iloc[:,5:]
    # print(point_A_coords.shape, point_B_coords.shape)
    for i in range(len(src_img_names)):
        src_fn= f'{path}/../{src_img_names[i]}'
        trg_fn= f'{path}/../{trg_img_names[i]}'
        src_size=Image.open(src_fn).size
        trg_size=Image.open(trg_fn).size

        if not split.startswith('train'):
            point_coords_src = get_points(point_A_coords, i).transpose(1,0)
            point_coords_trg = get_points(point_B_coords, i).transpose(1,0)
        else:
            src_anns = os.path.join(path, 'Annotations', category,
                                    os.path.basename(src_fn))[:-4] + '.mat'
            trg_anns = os.path.join(path, 'Annotations', category,
                                    os.path.basename(trg_fn))[:-4] + '.mat'
            point_coords_src = process_kps_pascal(read_mat(src_anns, 'kps'))
            point_coords_trg = process_kps_pascal(read_mat(trg_anns, 'kps'))

        # print(src_size)
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(point_coords_src, src_size[0], src_size[1], size)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(point_coords_trg, trg_size[0], trg_size[1], size)
        kps.append(source_kps)
        kps.append(target_kps)
        files.append(src_fn)
        files.append(trg_fn)
    
    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    # logger.info(f'Final number of used key points: {kps.size(1)}')
    return files, kps, None, used_kps
