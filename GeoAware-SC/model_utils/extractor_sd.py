import itertools
from contextlib import ExitStack
import torch
from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
import numpy as np
import torch.nn.functional as F
from detectron2.config import instantiate
from detectron2.data import MetadataCatalog
from detectron2.config import LazyCall as L
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.visualizer import ColorMode, random_color

from odise import model_zoo
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.data import get_openseg_labels
from odise.modeling.wrapper import OpenPanopticInference

COCO_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 1
]
COCO_THING_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
COCO_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 0
]
COCO_STUFF_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]

ADE_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
]
ADE_THING_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
ADE_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
]
ADE_STUFF_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]

LVIS_CLASSES = get_openseg_labels("lvis_1203", True)
# use beautiful coco colors
LVIS_COLORS = list(
    itertools.islice(itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(LVIS_CLASSES))
)


class StableDiffusionSeg(object):
    def __init__(self, model, metadata, aug, instance_mode=ColorMode.IMAGE):
        """
        Args:
            model (nn.Module):
            metadata (MetadataCatalog): image metadata.
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.model = model
        self.metadata = metadata
        self.aug = aug
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

    def get_features(self, original_image, caption=None, pca=None):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            features (dict):
                the output of the model for one image only.
        """
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = aug_input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        if caption is not None:
            features = self.model.get_features([inputs],caption,pca=pca)
        else:
            features = self.model.get_features([inputs],pca=pca)
        return features
    
    def predict(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = aug_input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions

def build_demo_classes_and_metadata(vocab, label_list):
    extra_classes = []

    if vocab:
        for words in vocab.split(";"):
            extra_classes.append([word.strip() for word in words.split(",")])
    extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]

    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []

    if "COCO" in label_list:
        demo_thing_classes += COCO_THING_CLASSES
        demo_stuff_classes += COCO_STUFF_CLASSES
        demo_thing_colors += COCO_THING_COLORS
        demo_stuff_colors += COCO_STUFF_COLORS
    if "ADE" in label_list:
        demo_thing_classes += ADE_THING_CLASSES
        demo_stuff_classes += ADE_STUFF_CLASSES
        demo_thing_colors += ADE_THING_COLORS
        demo_stuff_colors += ADE_STUFF_COLORS
    if "LVIS" in label_list:
        demo_thing_classes += LVIS_CLASSES
        demo_thing_colors += LVIS_COLORS

    MetadataCatalog.pop("odise_demo_metadata", None)
    demo_metadata = MetadataCatalog.get("odise_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }

    demo_classes = demo_thing_classes + demo_stuff_classes

    return demo_classes, demo_metadata

import sys


def load_model(config_path="Panoptic/odise_label_coco_50e.py", seed=42, diffusion_ver="v1-3", image_size=1024, num_timesteps=0, block_indices=(2,5,8,11), decoder_only=True, encoder_only=False, resblock_only=False):
    cfg = model_zoo.get_config(config_path, trained=True)

    cfg.model.backbone.feature_extractor.init_checkpoint = "sd://"+diffusion_ver
    cfg.model.backbone.feature_extractor.steps = (num_timesteps,)
    cfg.model.backbone.feature_extractor.unet_block_indices = block_indices
    cfg.model.backbone.feature_extractor.encoder_only = encoder_only
    cfg.model.backbone.feature_extractor.decoder_only = decoder_only
    cfg.model.backbone.feature_extractor.resblock_only = resblock_only
    cfg.model.overlap_threshold = 0
    seed_all_rng(seed)

    cfg.dataloader.test.mapper.augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=image_size, sample_style="choice", max_size=2560),
        ]
    dataset_cfg = cfg.dataloader.test

    aug = instantiate(dataset_cfg.mapper).augmentations

    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load(cfg.train.init_checkpoint)

    return model, aug

def inference(model, aug, image, vocab, label_list):

    demo_classes, demo_metadata = build_demo_classes_and_metadata(vocab, label_list)
    with ExitStack() as stack:
        inference_model = OpenPanopticInference(
            model=model,
            labels=demo_classes,
            metadata=demo_metadata,
            semantic_on=False,
            instance_on=False,
            panoptic_on=True,
        )
        stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())

        demo = StableDiffusionSeg(inference_model, demo_metadata, aug)
        pred = demo.predict(np.array(image))
        return (pred, demo_classes)
    
def get_features(model, aug, image, vocab, label_list, caption=None, pca=False):
    
    demo_classes, demo_metadata = build_demo_classes_and_metadata(vocab, label_list)
    with ExitStack() as stack:
        inference_model = OpenPanopticInference(
            model=model,
            labels=demo_classes,
            metadata=demo_metadata,
            semantic_on=False,
            instance_on=False,
            panoptic_on=True,
        )
        stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())

        demo = StableDiffusionSeg(inference_model, demo_metadata, aug)
        if caption is not None:
            features = demo.get_features(np.array(image), caption, pca=pca)
        else:
            features = demo.get_features(np.array(image), pca=pca)
        return features


def process_features_and_mask(model, aug, image, category=None, input_text=None, mask=False, raw=True):

    input_image = image
    caption = input_text
    vocab = ""
    label_list = ["COCO"]
    category_convert_dict={
        'aeroplane':'airplane',
        'motorbike':'motorcycle',
        'pottedplant':'potted plant',
        'tvmonitor':'tv',
    }
    if type(category) is not list and category in category_convert_dict:
        category=category_convert_dict[category]
    elif type(category) is list:
        category=[category_convert_dict[cat] if cat in category_convert_dict else cat for cat in category]
    features = get_features(model, aug, input_image, vocab, label_list, caption, pca=raw)
    return features

def get_mask(model, aug, image, category=None, input_text=None):
    model.backbone.feature_extractor.decoder_only = False
    model.backbone.feature_extractor.encoder_only = False
    model.backbone.feature_extractor.resblock_only = False
    input_image = image
    vocab = ""
    label_list = ["COCO"]
    category_convert_dict={
        'aeroplane':'airplane',
        'motorbike':'motorcycle',
        'pottedplant':'potted plant',
        'tvmonitor':'tv',
    }
    if type(category) is not list and category in category_convert_dict:
        category=category_convert_dict[category]
    elif type(category) is list:
        category=[category_convert_dict[cat] if cat in category_convert_dict else cat for cat in category]

    (pred,classes) =inference(model, aug, input_image, vocab, label_list)
    seg_map=pred['panoptic_seg'][0]
    target_mask_id = []
    for item in pred['panoptic_seg'][1]:
        item['category_name']=classes[item['category_id']]
        if type(category) is list:
            for cat in category:
                if cat in item['category_name']:
                    target_mask_id.append(item['id'])
        else:
            if category in item['category_name']:
                target_mask_id.append(item['id'])
    resized_seg_map_s4 = seg_map.float()
    binary_seg_map = torch.zeros_like(resized_seg_map_s4)
    for i in target_mask_id:
        binary_seg_map += (resized_seg_map_s4 == i).float()
    if len(target_mask_id) == 0 or binary_seg_map.sum() < 6:
        binary_seg_map = torch.ones_like(resized_seg_map_s4)

    return binary_seg_map
