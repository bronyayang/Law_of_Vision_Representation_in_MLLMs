import torch
import numpy as np
import pickle
import json
from tqdm import tqdm
from loguru import logger
import os
import sys
from utils.utils_geoware import (
    SPAIR_GEO_AWARE,
    SPAIR_FLIP,
)
from utils.utils_dataset import preprocess_kps_pad

def load_spair_data(path1, path2, size=256):
    source_kps = torch.zeros(30, 3)
    target_kps = torch.zeros(30, 3)

    source_fn = path1
    target_fn = path2
    source_json_name = source_fn.replace("JPEGImages", "ImageAnnotation").replace(
        "jpg", "json"
    )
    target_json_name = target_fn.replace("JPEGImages", "ImageAnnotation").replace(
        "jpg", "json"
    )

    with open(source_json_name) as f:
        file = json.load(f)
        azimuth_src = file["azimuth_id"]
        kpts_src = file["kps"]
        source_bbox = np.asarray(file["bndbox"])
        source_size = (file["image_width"], file["image_height"])  # (W, H)
    with open(target_json_name) as f:
        file = json.load(f)
        azimuth_trg = file["azimuth_id"]
        kpts_trg = file["kps"]
        target_bbox = np.asarray(file["bndbox"])
        target_size = (file["image_width"], file["image_height"])  # (W, H)

    for i in range(30):
        point = kpts_src[str(i)]
        if point is None:
            source_kps[i, :3] = 0
        else:
            source_kps[i, :2] = torch.Tensor(point).float()     # set xy
            source_kps[i, 2] = 1                                # set visibility

    source_kps, src_x, src_y, src_scale = preprocess_kps_pad(
        source_kps, source_size[0], source_size[1], size
    )

    for i in range(30):
        point = kpts_trg[str(i)]
        if point is None:
            target_kps[i, :3] = 0
        else:
            target_kps[i, :2] = torch.Tensor(point).float()
            target_kps[i, 2] = 1

    target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(
        target_kps, target_size[0], target_size[1], size
    )

    thresholds = (
        max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0])
        * trg_scale
    )

    if (azimuth_src <= 3 and azimuth_trg > 3) or (azimuth_trg <= 3 and azimuth_src > 3):
        mirror = 1
    else:
        mirror = 0


    return source_kps, target_kps, thresholds, azimuth_src, azimuth_trg, mirror

def load_ap10k_data(path1, path2, size=256):
    src_json_path = path1.replace("JPEGImages", "ImageAnnotation").replace(
        "jpg", "json"
    )
    trg_json_path = path2.replace("JPEGImages", "ImageAnnotation").replace(
        "jpg", "json"
    )
    with open(src_json_path) as f:
        src_file = json.load(f)
    with open(trg_json_path) as f:
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
    
    thresholds=(max(target_bbox[3], target_bbox[2])*trg_scale)

    return source_kps, target_kps, thresholds

def print_pck_transfer(log_path, target="PCK-Transfer@0.10:"):
    pck_10=[]
    with open(log_path, 'r') as f:
        for line in f:
            if target in line:
                pck_transfer_value_10 = line.strip().split(target)[1].split("%")[0].strip()
                pck_10.append(float(pck_transfer_value_10))
                print(pck_transfer_value_10)

# len(result)
# print(result[0].keys())
def convert_all_results(result):
    all_results = []
    for item in tqdm(result[:]):
        src_fn = item["src_fn"]             #must
        tgt_fn = item["trg_fn"]             #must
        category = src_fn.split("/")[-2]
        reso = item["resize_resolution"]    #must
        pred_kps = item["src_kpts_pred"]    #must
        # convert to tensor
        pred_kps = torch.tensor(pred_kps[:,[1,0]]).float()
        src_kps, trg_kps, thresholds, src_az, trg_az, mirror = load_spair_data(
            src_fn, tgt_fn, reso
        )
        
        az = min(abs(src_az - trg_az), 8 - abs(src_az - trg_az))
        # convert thresholds to tensor
        thresholds = torch.tensor(thresholds).float()

        vis = src_kps[:, 2] * trg_kps[:, 2] > 0
        vis1 = src_kps[:, 2] > 0
        vis2 = trg_kps[:, 2] > 0

        trg_kps = trg_kps[:, [1, 0]]
        src_kps = src_kps[:, [1, 0]]

        used_points = [idx for idx in range(30) if vis[idx]]
        influ_list_geo = SPAIR_GEO_AWARE[category]
        influ_list_flip = SPAIR_FLIP[category]
        geo_aware_list = []

        for item in influ_list_geo:
            # convert to list
            item = [item] if isinstance(item, int) else item
            # check if all items are visible
            temp_list = [idx for idx in item if vis[idx]]
            temp_list2 = [idx for idx in item if vis2[idx]]
            if len(temp_list2) >= 2 and len(temp_list) >= 1:
                for temp_idx in temp_list:
                    geo_aware_list.append([temp_idx])
        geo_aware_idx = [item for sublist in geo_aware_list for item in sublist]

        # flip
        flip_list = []
        for item in influ_list_flip:
            # convert to list
            item = [item] if isinstance(item, int) else item
            # check if all items are visible
            temp_list = [idx for idx in item if vis[idx]]
            temp_list1 = [idx for idx in item if vis1[idx]]
            if len(temp_list1) >= 2 and len(temp_list) >= 1:
                for temp_idx in temp_list:
                    flip_list.append([temp_idx])
        flip_idx = [item for sublist in flip_list for item in sublist]

        all_results.append(
            {   
                "src_kps": src_kps,
                "gt_kps": trg_kps,
                "src_fn": src_fn,
                "trg_fn": tgt_fn,
                "pred_kps": pred_kps,
                "thresholds": thresholds,
                "category": category,
                "az": az,
                "mirror": mirror,
                "geo_aware_idx": geo_aware_idx,
                "flip_idx": flip_idx,
                "used_points": used_points,
                # "new_used_points": new_used_points,
            }
        )
    return all_results

def convert_all_results_cats(result):
    all_results = []
    for item in tqdm(result[:]):
        src_fn = item["src_fn"]             #must
        tgt_fn = item["trg_fn"]             #must
        category = src_fn.split("/")[-2]
        reso = item["resize_resolution"]    #must
        pred_kps = item["src_kpts_pred"]    #must
        # convert to tensor
        pred_kps = torch.tensor(pred_kps[:,[1,0]]).float()
        src_kps, trg_kps, _, src_az, trg_az, mirror = load_spair_data(
            src_fn, tgt_fn, reso
        )
        
        az = min(abs(src_az - trg_az), 8 - abs(src_az - trg_az))
        # convert thresholds to tensor
        thresholds = item["threshoulds"]

        vis = src_kps[:, 2] * trg_kps[:, 2] > 0
        vis1 = src_kps[:, 2] > 0
        vis2 = trg_kps[:, 2] > 0

        trg_kps = item["trg_kpts"][:, [1,0]]

        used_points = [idx for idx in range(30) if vis[idx]]
        influ_list_geo = SPAIR_GEO_AWARE[category]
        influ_list_flip = SPAIR_FLIP[category]
        geo_aware_list = []

        for item in influ_list_geo:
            # convert to list
            item = [item] if isinstance(item, int) else item
            # check if all items are visible
            temp_list = [idx for idx in item if vis[idx]]
            temp_list2 = [idx for idx in item if vis2[idx]]
            if len(temp_list2) >= 2 and len(temp_list) >= 1:
                for temp_idx in temp_list:
                    geo_aware_list.append([temp_idx])
        geo_aware_idx = [item for sublist in geo_aware_list for item in sublist]

        # flip
        flip_list = []
        for item in influ_list_flip:
            # convert to list
            item = [item] if isinstance(item, int) else item
            # check if all items are visible
            temp_list = [idx for idx in item if vis[idx]]
            temp_list1 = [idx for idx in item if vis1[idx]]
            if len(temp_list1) >= 2 and len(temp_list) >= 1:
                for temp_idx in temp_list:
                    flip_list.append([temp_idx])
        flip_idx = [item for sublist in flip_list for item in sublist]

        all_results.append(
            {   
                "gt_kps": trg_kps,
                "pred_kps": pred_kps,
                "thresholds": thresholds,
                "category": category,
                "az": az,
                "mirror": mirror,
                "geo_aware_idx": geo_aware_idx,
                "flip_idx": flip_idx,
                "used_points": used_points,
                # "new_used_points": new_used_points,
            }
        )
    return all_results

def convert_all_results_ap10k(result):
    all_results = []
    for item in tqdm(result[:]):
        src_fn = item["src_fn"]             #must
        tgt_fn = item["trg_fn"]             #must
        category = src_fn.split("/")[-2]
        reso = item["resize_resolution"]    #must
        pred_kps = item["src_kpts_pred"]    #must
        # convert to tensor
        pred_kps = torch.tensor(pred_kps[:,[1,0]]).float()
        src_kps, trg_kps, thresholds = load_ap10k_data(
            src_fn, tgt_fn, reso
        )
        thresholds = torch.tensor(thresholds).float()

        vis = src_kps[:, 2] * trg_kps[:, 2] > 0
        vis1 = src_kps[:, 2] > 0
        vis2 = trg_kps[:, 2] > 0

        trg_kps = trg_kps[:, [1, 0]]
        src_kps = src_kps[:, [1, 0]]

        used_points = [idx for idx in range(17) if vis[idx]]

        influ_list_geo = [0,1, # eye
                                2, # nose
                                3, # neck
                                4, # root of tail
                                [5,8], # shoulder
                                [6,9,12,15], # elbow # knee
                                [7,10,13,16], # front paw # back paw
                                [11,14], # hip
                              ]
        geo_aware_list = []
        for item in influ_list_geo:
            # convert to list
            item = [item] if isinstance(item, int) else item
            # check if all items are visible
            temp_list = [idx for idx in item if vis[idx]]
            temp_list2 = [idx for idx in item if vis2[idx]]
            if len(temp_list2) >= 2 and len(temp_list) >= 1:
                for temp_idx in temp_list:
                    geo_aware_list.append([temp_idx])
        geo_aware_idx = [item for sublist in geo_aware_list for item in sublist]

        all_results.append(
            {   
                "src_kps": src_kps,
                "gt_kps": trg_kps,
                "pred_kps": pred_kps,
                "thresholds": thresholds,
                "category": category,
                "used_points": used_points,
                "geo_aware_idx": geo_aware_idx,
            }
        )
    return all_results

def get_std_result(all_results, cls=None, geo=False, flip=False, az=None):
    all_gt = []
    all_pred = []
    all_thresholds = []
    for item in all_results:
        if (cls is None or item["category"] == cls) and (az is None or item["az"] == az):
            gt_kps = item["gt_kps"]
            pred_kps = item["pred_kps"]
            used_points = item["used_points"]
            thresholds = item["thresholds"]
            if geo or flip:
                geo_idx = item["geo_aware_idx"] if geo else item["flip_idx"]
                all_gt.append(gt_kps[geo_idx])
                all_pred.append(pred_kps[geo_idx])
                all_thresholds.append(thresholds.repeat(len(geo_idx)))
            else:
                all_gt.append(gt_kps[used_points])
                all_pred.append(pred_kps[used_points])
                all_thresholds.append(thresholds.repeat(len(used_points)))
    all_gt = torch.cat(all_gt)
    all_pred = torch.cat(all_pred)
    all_thresholds = torch.cat(all_thresholds)
    alpha = torch.tensor([0.1, 0.05, 0.01])
    all_thresholds = alpha.unsqueeze(1) * all_thresholds.unsqueeze(0)

    err = torch.abs(all_gt - all_pred).norm(dim=-1)
    correct = err.unsqueeze(0).repeat(len(alpha), 1) < all_thresholds
    correct = correct.sum(dim=-1).float() / len(all_gt)
    
    return correct, len(all_gt)

def get_img_result(all_results, cls=None, geo=False, flip=False, az=None):
    all_correct = []
    geo_img = 0
    for item in all_results:
        if (cls is None or item["category"] == cls) and (az is None or item["az"] == az):
            gt_kps = item["gt_kps"]
            pred_kps = item["pred_kps"]
            used_points = item["used_points"]
            thresholds = item["thresholds"]
            alpha = torch.tensor([0.1, 0.05, 0.01])
            if geo or flip:
                geo_idx = item["geo_aware_idx"] if geo else item["flip_idx"]
                if len(geo_idx) == 0:
                    continue
                gt = gt_kps[geo_idx]
                pred = pred_kps[geo_idx]
                thresholds = alpha.unsqueeze(1) * thresholds.repeat(len(geo_idx)).unsqueeze(0)
                geo_img += 1
            else:
                gt = gt_kps[used_points]
                pred = pred_kps[used_points]
                thresholds = alpha.unsqueeze(1) * thresholds.repeat(len(used_points)).unsqueeze(0)
            err = torch.abs(gt - pred).norm(dim=-1)
            correct = err.unsqueeze(0).repeat(len(alpha), 1) < thresholds
            correct = correct.sum(dim=-1).float() / len(gt)
            all_correct.append(correct)
    
    if len(all_correct) == 0:
        return torch.zeros(3), 0
    
    all_correct = torch.stack(all_correct, dim=0)
    # get avg result for each image
    avg_correct = torch.mean(all_correct, dim=0)
    if not geo:
        return avg_correct, len(all_correct)
    else:
        return avg_correct, geo_img

if __name__ == "__main__":
    from utils.logger import get_logger
    try:
        # get the input_path from the command line
        input_path = sys.argv[1]
    except:
        input_path = "results_spair/pck_train__sample_1_None_0_lr_0.0005_DIM_768_BBOX_True_sd_True_dino_True_GEO_0/result.pkl"
    
    try:
        dataset = sys.argv[2]
    except:
        if "results_ap10k" in input_path:
            dataset = "ap10k"
        elif "results_pascal" in input_path:
            dataset = "pascal"
        else:
            dataset = "spair"

    assert dataset in ["spair", "pascal", "ap10k"]
    with open(input_path, "rb") as f:
            result = pickle.load(f)
    cls = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'train', 'tvmonitor']
    if dataset == "spair":
        if "cats" in input_path:
            all_results = convert_all_results_cats(result)
        else:
            all_results = convert_all_results(result)
    elif dataset == "ap10k":
        all_results = convert_all_results_ap10k(result)

    az = None
    logger_path = input_path.replace("result.pkl","metric.log")
    while os.path.exists(logger_path):
        logger_path = logger_path.replace('.log', '1.log')
    logger = get_logger(logger_path)

    if dataset == "spair":
        # compute per kpt result
        avg = []
        avg_geo = []
        for cls_name in cls:
            std_result, _ = get_img_result(all_results, cls_name, az=az)
            std_result_geo, _ = get_img_result(all_results, cls_name, geo=True, az=az)
            # print(cls_name, std_result[0], std_result_geo[0])
            logger.info(f'{cls_name}.....std_PCK@0.10: {std_result[0]*100:.2f}%, geo_PCK@0.10: {std_result_geo[0]*100:.2f}%')
            avg.append(std_result[0])
            avg_geo.append(std_result_geo[0])
        # print(np.mean(avg))
    std_result, len_std = get_std_result(all_results, az=az)
    img_result, len_img = get_img_result(all_results, az=az)
    if dataset == "spair":
        std_result_geo, len_geo = get_img_result(all_results, geo=True, az=az)
    elif dataset == "ap10k":
        std_result_geo, len_geo = get_std_result(all_results, geo=True, az=az)
    logger.info(std_result)
    logger.info(img_result)
    logger.info(std_result_geo)
    if dataset == "spair":
        print_pck_transfer(logger_path, target="std_PCK@0.10:")
        print()
        print_pck_transfer(logger_path, target="geo_PCK@0.10:")



    """get the result of sweeping all azs"""
    # for az in range(5):
        
    #     logger_path = input_path.replace("result.pkl","metric.log")
    #     while os.path.exists(logger_path):
    #         logger_path = logger_path.replace('.log', '1.log')
    #     logger = get_logger(logger_path)

    #     if dataset == "spair":
    #         # compute per kpt result
    #         avg = []
    #         avg_geo = []
    #         for cls_name in cls:
    #             std_result, _ = get_img_result(all_results, cls_name, az=az)
    #             std_result_geo, _ = get_img_result(all_results, cls_name, geo=True, az=az)
    #             # print(cls_name, std_result[0], std_result_geo[0])
    #             logger.info(f'{cls_name}.....std_PCK@0.10: {std_result[0]*100:.2f}%, geo_PCK@0.10: {std_result_geo[0]*100:.2f}%')
    #             avg.append(std_result[0])
    #             avg_geo.append(std_result_geo[0])
    #         # print(np.mean(avg))
    #     std_result, _ = get_std_result(all_results, az=az)
    #     img_result, _ = get_img_result(all_results, az=az)
    #     if dataset == "spair":
    #         std_result_geo, _ = get_img_result(all_results, geo=True, az=az)
    #     elif dataset == "ap10k":
    #         std_result_geo, _ = get_std_result(all_results, geo=True, az=az)
    #     logger.info(std_result)
    #     logger.info(img_result)
    #     logger.info(std_result_geo)
    #     if dataset == "spair":
    #         print_pck_transfer(logger_path, target="std_PCK@0.10:")
    #         print()
    #         print_pck_transfer(logger_path, target="geo_PCK@0.10:")



    """get the result of all pcks"""
    # avg = []
    # avg_geo = []
    # for cls_name in cls:
    #     std_result = get_img_result(all_results, cls_name)
    #     std_result_geo = get_img_result(all_results, cls_name, geo=True)
    #     # print(cls_name, std_result[0], std_result_geo[0])
    #     avg.append(std_result[0])
    #     avg_geo.append(std_result_geo[0])
    # # print(np.mean(avg))
    # print(np.mean(avg_geo))

    # std_result = get_img_result(all_results)
    # std_result_geo = get_img_result(all_results, geo=True)
    # # print(std_result)
    # # print(std_result_geo)


    # def get_std_result_special(all_results, cls=None, geo=False, flip=False, special_filter=None):
    #     all_gt = []
    #     all_pred = []
    #     all_thresholds = []
    #     for item in all_results:
    #         if cls is None or item["category"] == cls:
    #             gt_kps = item["gt_kps"]
    #             pred_kps = item["pred_kps"]
    #             used_points = item["used_points"]
    #             thresholds = item["thresholds"]
    #             geo_idx = item["geo_aware_idx"] if geo else item["flip_idx"]
    #             if special_filter is not None:
    #                 geo_idx = special_filter if special_filter[0] in used_points else []
    #             if geo or flip or special_filter is not None:
    #                 all_gt.append(gt_kps[geo_idx])
    #                 all_pred.append(pred_kps[geo_idx])
    #                 all_thresholds.append(thresholds.repeat(len(geo_idx)))
    #             else:
    #                 all_gt.append(gt_kps[used_points])
    #                 all_pred.append(pred_kps[used_points])
    #                 all_thresholds.append(thresholds.repeat(len(used_points)))
    #     all_gt = torch.cat(all_gt)
    #     all_pred = torch.cat(all_pred)
    #     all_thresholds = torch.cat(all_thresholds)
    #     alpha = torch.tensor([0.1, 0.05, 0.01])
    #     all_thresholds = alpha.unsqueeze(1) * all_thresholds.unsqueeze(0)

    #     err = torch.abs(all_gt - all_pred).norm(dim=-1)
    #     correct = err.unsqueeze(0).repeat(len(alpha), 1) < all_thresholds
    #     correct = correct.sum(dim=-1).float() / len(all_gt)
        
    #     return correct

    # # compute per kpt result
    # for cls_name in cls:
    #     cls_result = []
    #     for idx in range(30):
    #         special_filter = [idx]
    #         std_result = get_std_result_special(all_results, cls_name, special_filter=special_filter)
    #         cls_result.append({idx:std_result[0].item()})
    #     # sort the cls_result according to value
    #     cls_result = sorted(cls_result, key=lambda x: list(x.values()), reverse=True)
    #     formatted_results = ', '.join([f"{k}:{v*100:.2f}" for item in cls_result for k, v in item.items()])
    #     print(f"{cls_name}.....:{formatted_results}")
