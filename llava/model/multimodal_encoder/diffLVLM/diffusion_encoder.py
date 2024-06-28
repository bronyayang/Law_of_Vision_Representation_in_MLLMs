import torch
import torch.nn as nn

import argparse
from PIL import Image
from torchvision.transforms import PILToTensor
from .src.models.dift_sd import SDFeaturizer
from .src.models.dift_imsd import IMSDFeaturizer
from .src.models.dift_dit import DiTFeaturizer
from .src.models.dift_sd3 import SD3Featurizer
from typing import Dict, List, Optional, Union
from transformers import AutoConfig

build_featurelizer_mapping = {'lambdalabs/sd-image-variations-diffusers': IMSDFeaturizer,
                               'stabilityai/stable-diffusion-2-1': SDFeaturizer,
                               'runwayml/stable-diffusion-v1-5': SDFeaturizer,
                               'stabilityai/stable-diffusion-xl-base-1.0': SDFeaturizer,
                               'facebook/DiT-XL-2-512': DiTFeaturizer,
                               'stabilityai/stable-diffusion-3-medium-diffusers': SD3Featurizer}


feature_hid_size_mapping = {'runwayml/stable-diffusion-v1-5_feature': 1280,
                            'lambdalabs/sd-image-variations-diffusers': 1280,
                            'runwayml/stable-diffusion-v1-5': 1280,
                            'stabilityai/stable-diffusion-xl-base-1.0': 1280,
                            'stabilityai/stable-diffusion-2-1': 1280,
                            'facebook/DiT-XL-2-512': 4608,
                            'stabilityai/stable-diffusion-3-medium-diffusers': 6144}

class DiffImageProcessor(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.crop_size = {'height':img_size[0], 'width':img_size[1]}
        
    def preprocess(self, img, return_tensors: Optional[str] = None, **kwargs):
        if self.img_size[0] > 0:
            img = img.resize(self.img_size)
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        data = {"pixel_values": [img_tensor]}
        return data


class DiffVisionTower(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.is_loaded = False
        # add new args
        self.up_ft_index = args.up_ft_index
        self.t = args.t
        self.prompt = args.prompt
        self.model_id = args.vision_tower
        self.ensemble_size = args.ensemble_size
        self.img_size = [args.img_size, args.img_size]
        self.hidden_size_num = feature_hid_size_mapping[args.vision_tower]

        self.load_model()
        

    def load_model(self):
        # add new image processor
        self.image_processor = DiffImageProcessor(self.img_size)
        self.vision_tower = build_featurelizer_mapping[self.model_id](self.model_id)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower.forward(image,
                                                          prompt=self.prompt,
                                                          t=self.t,
                                                          up_ft_index=self.up_ft_index,
                                                          ensemble_size=self.ensemble_size)
                image_features.append(image_feature)
        else:
            if len(images.shape) == 3:
                images = torch.unsqueeze(images, dim=0)
            image_features = self.vision_tower.forward(images,
                                            prompt=self.prompt,
                                            t=self.t,
                                            up_ft_index=self.up_ft_index,
                                            ensemble_size=self.ensemble_size)

        if len(image_features.shape) == 3:
            image_features = torch.unsqueeze(image_features, dim=0)
        image_features = image_features.permute(0, 2, 3, 1)
        B, H, W, C = image_features.shape
        image_features = image_features.view(B, -1, C)
        return image_features

    # @property
    # def dummy_feature(self):
    #     return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype
    
    @property
    def device(self):
        return self.vision_tower.device

    # @property
    # def config(self):
    #     if self.is_loaded:
    #         return self.vision_tower.config
    #     else:
    #         return self.cfg_only

    @property
    def hidden_size(self):
        return self.hidden_size_num

    # @property
    # def num_patches(self):
    #     return (self.config.image_size // self.config.patch_size) ** 2