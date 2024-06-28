import os
from .clip_encoder import CLIPVisionTower
from .diffLVLM.diffusion_encoder import DiffVisionTower
from .dinov2_encoder import DinoV2VisionTower
from .siglip_encoder import SigLipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_diffusion_vision_tower(vision_tower_cfg, **kwargs):
    # vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # is_absolute_path_exists = os.path.exists(vision_tower)
    return DiffVisionTower(args=vision_tower_cfg)

def build_dinov2_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    return DinoV2VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

def build_siglip_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    return SigLipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

def build_feature(vision_tower_cfg, **kwargs):
    return 'feature'