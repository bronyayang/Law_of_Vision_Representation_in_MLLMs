import sys
import os
import yaml
import numpy as np
from loguru import logger
from utils.eval_spair import get_img_result, convert_all_results

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_logger(output_file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    while os.path.exists(output_file):
        output_file = output_file.replace('.log', '1.log')
    if output_file:
        logger.add(output_file, enqueue=True, format=log_format)
    return logger

def update_stats(args, pcks, pcks_05, pcks_01, weights, kpt_weights, pck, img_correct):
    """
    Update the lists for PCK statistics.

    Parameters:
    - pcks, pcks_05, pcks_01: Lists to hold PCK scores at different thresholds.
    - weights, kpt_weights: Lists to hold weights for averaging the PCK scores.
    - pck: Tuple containing PCK scores and weights.
    - img_correct: Tuple containing image correctness scores and weights.
    """
    if args.KPT_RESULT:
        pcks.append(pck[0])
        pcks_05.append(pck[1])
        pcks_01.append(pck[2])
        weights.append(pck[3])
    else:
        pcks.append(img_correct[0])
        pcks_05.append(img_correct[1])
        pcks_01.append(img_correct[2])
        weights.append(img_correct[3])
    kpt_weights.append(pck[3])

def update_geo_stats(geo_aware, geo_aware_count, pcks_geo, pcks_geo_05, pcks_geo_01, weights_geo, correct_geo):
    """
    Update the lists for geo-aware statistics.

    Parameters:
    - geo_aware, geo_aware_count: Lists to hold geo-aware counts.
    - pcks_geo, pcks_geo_05, pcks_geo_01: Lists to hold geo-aware PCK scores at different thresholds.
    - weights_geo: List to hold weights for averaging the geo-aware PCK scores.
    - correct_geo: Tuple containing geo-aware correctness scores and counts.
    """
    geo_aware.append(correct_geo[0])
    geo_aware_count.append(correct_geo[1])
    pcks_geo.append(correct_geo[2])
    pcks_geo_05.append(correct_geo[3])
    pcks_geo_01.append(correct_geo[4])
    weights_geo.append(correct_geo[5])

def log_weighted_pcks(args, logger, pcks, pcks_05, pcks_01, weights):
    """Logs weighted PCK statistics and returns PCK values."""
    pck_010 = np.average(pcks, weights=weights)
    pck_005 = np.average(pcks_05, weights=weights)
    pck_001 = np.average(pcks_01, weights=weights)

    if not args.KPT_RESULT and args.TRAIN_DATASET == "spair":  # Image result
        logger.info(f"Weighted Per image PCK0.10: {pck_010 * 100:.2f}%, image PCK0.05: {pck_005 * 100:.2f}%, image PCK0.01: {pck_001 * 100:.2f}%")
    else:
        logger.info(f"Weighted Per kpt PCK0.10: {pck_010 * 100:.2f}%, kpt PCK0.05: {pck_005 * 100:.2f}%, kpt PCK0.01: {pck_001 * 100:.2f}")
    
    return pck_010, pck_005, pck_001


def log_geo_stats(args, geo_aware, geo_aware_count, pcks_geo, pcks_geo_05, pcks_geo_01, weights_geo, weights_kpt, total_out_results):
    """
    Log the geo-aware statistics.

    Parameters:
    - geo_aware, geo_aware_count: Lists to hold geo-aware counts.
    - pcks_geo, pcks_geo_05, pcks_geo_01: Lists to hold geo-aware PCK scores at different thresholds.
    - weights_geo: Weights for averaging the geo-aware PCK scores.
    """
    avg_geo_aware = np.average(geo_aware) * 100
    avg_geo_aware_count = np.average(geo_aware_count, weights=weights_kpt) * 100

    logger.info(f"Average images geo-aware occurrence: {avg_geo_aware:.2f}%, Average points geo-aware occurrence: {avg_geo_aware_count:.2f}%")
    
    if not args.KPT_RESULT and args.TRAIN_DATASET == "spair":  # Image result
        converted_total_results = convert_all_results(total_out_results)
        avg_pck_geo_010, avg_pck_geo_005, avg_pck_geo_001 = get_img_result(converted_total_results, geo=True)[0].tolist()
        logger.info(f"Weighted Per image geo-aware PCK0.10: {avg_pck_geo_010*100:.2f}%, image PCK0.05: {avg_pck_geo_005*100:.2f}%, image PCK0.01: {avg_pck_geo_001*100:.2f}%")
    else:
        avg_pck_geo_010 = np.average(pcks_geo, weights=weights_geo) * 100
        avg_pck_geo_005 = np.average(pcks_geo_05, weights=weights_geo) * 100
        avg_pck_geo_001 = np.average(pcks_geo_01, weights=weights_geo) * 100
        logger.info(f"Weighted Per kpts geo-aware PCK0.10: {avg_pck_geo_010:.2f}%, kpts PCK0.05: {avg_pck_geo_005:.2f}%, kpts PCK0.01: {avg_pck_geo_001:.2f}%")
    