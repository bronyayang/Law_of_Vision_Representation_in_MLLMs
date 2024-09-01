#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .multimodal_encoder.builder import build_vision_tower, build_diffusion_vision_tower, build_dinov2_vision_tower, build_feature, build_siglip_vision_tower
from .multimodal_projector.builder import build_vision_projector
from llava.mm_utils import get_anyres_image_grid_shape

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

build_function_mapping = {'openai/clip-vit-large-patch14-336': build_vision_tower, 
                        'google/siglip-base-patch16-224': build_siglip_vision_tower, 
                        'laion/CLIP-ViT-L-14-laion2B-s32B-b82K': build_vision_tower,
                        'stabilityai/stable-diffusion-2-1': build_diffusion_vision_tower,
                        'runwayml/stable-diffusion-v1-5': build_diffusion_vision_tower,
                        'lambdalabs/sd-image-variations-diffusers': build_diffusion_vision_tower,
                        'facebook/dinov2-large': build_dinov2_vision_tower,
                        'stabilityai/stable-diffusion-xl-base-1.0': build_diffusion_vision_tower,
                        'feature': build_feature,
                        'facebook/DiT-XL-2-512': build_diffusion_vision_tower,
                        'stabilityai/stable-diffusion-3-medium-diffusers': build_diffusion_vision_tower,
                        'openai/clip-vit-large-patch14': build_vision_tower}

feature_hid_size_mapping = {'runwayml/stable-diffusion-v1-5_feature': 1280}

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            if ',' not in config.mm_vision_tower and '.' not in config.mm_vision_tower:
                if 'feature' in config.mm_vision_tower:
                    self.vision_tower = build_function_mapping['feature'](config)
                else:
                    self.vision_tower = build_function_mapping[config.mm_vision_tower](config)
            elif ',' in config.mm_vision_tower:
                # for mof
                ori_mm_vision_tower = config.mm_vision_tower
                vision_tower_name_split = config.mm_vision_tower.split(',')
                list_vision_towers = nn.ModuleList([])
                for v in vision_tower_name_split:
                    config.mm_vision_tower = v
                    if 'diffusion' in v:
                        config.vision_tower = v
                    if 'noise' in v:
                        list_vision_towers.append(None)
                    else:
                        list_vision_towers.append(build_function_mapping[v](config))
                self.vision_tower = list_vision_towers
                config.mm_vision_tower = ori_mm_vision_tower
            else:
                # feature fuse
                ori_mm_vision_tower = config.mm_vision_tower
                vision_tower_name_split = config.mm_vision_tower.split('.')
                list_vision_towers = nn.ModuleList([])
                for v in vision_tower_name_split:
                    config.mm_vision_tower = v
                    if 'diffusion' in v:
                        config.vision_tower = v
                    if 'noise' in v:
                        list_vision_towers.append(None)
                    else:
                        list_vision_towers.append(build_function_mapping[v](config))
                self.vision_tower = list_vision_towers
                config.mm_vision_tower = ori_mm_vision_tower

            if ',' not in config.mm_vision_tower:
                if 'feature' in config.mm_vision_tower:
                    config.mm_hidden_size = feature_hid_size_mapping[config.mm_vision_tower]
                self.mm_projector = build_vision_projector(config)
            else:
                # for now, mof only inplemented same projector type
                self.mm_projector = nn.ModuleList([])
                list_mm_hidden_size = config.mm_hidden_size
                config.mm_hidden_size = 0
                for i, v in enumerate(self.vision_tower):
                    config.mm_hidden_size += list_mm_hidden_size[i]
                    if v is None:
                        self.mm_projector.append(None)
                    else:
                        self.mm_projector.append(build_vision_projector(config))

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        # if type(vision_tower) is list:
        #     vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower_name = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        if ',' not in vision_tower_name and '.' not in vision_tower_name:
            self.config.mm_vision_tower = vision_tower_name
            if 'feature' in vision_tower_name:
                vision_tower = build_function_mapping['feature'](model_args)
            else:
                vision_tower = build_function_mapping[vision_tower_name](model_args)
        elif ',' in vision_tower_name:
            # for mof
            vision_tower_name_split = vision_tower_name.split(',')
            list_vision_towers = nn.ModuleList([])
            for v in vision_tower_name_split:
                if v == 'noise':
                    list_vision_towers.append(None)
                else:
                    self.config.mm_vision_tower = v
                    model_args.vision_tower = v
                    list_vision_towers.append(build_function_mapping[v](model_args))
            vision_tower = list_vision_towers
            self.config.mm_vision_tower = vision_tower_name
            model_args.vision_tower = vision_tower_name
        else:
            vision_tower_name_split = vision_tower_name.split('.')
            list_vision_towers = nn.ModuleList([])
            for v in vision_tower_name_split:
                if v == 'noise':
                    list_vision_towers.append(None)
                else:
                    self.config.mm_vision_tower = v
                    model_args.vision_tower = v
                    list_vision_towers.append(build_function_mapping[v](model_args))
            vision_tower = list_vision_towers
            self.config.mm_vision_tower = vision_tower_name
            model_args.vision_tower = vision_tower_name

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if ',' not in vision_tower_name and '.' not in vision_tower_name:
            if 'feature' not in vision_tower_name:
                self.config.mm_hidden_size = vision_tower.hidden_size
            else:
                self.config.mm_hidden_size = feature_hid_size_mapping[vision_tower_name]
            self.mm_projector = build_vision_projector(self.config)
        elif '.' in vision_tower_name:
            self.config.mm_hidden_size = 0
            for v in vision_tower:
                self.config.mm_hidden_size += v.hidden_size
            self.mm_projector = build_vision_projector(self.config)
        else:
            # for now, mof only inplemented same projector type
            self.mm_projector = nn.ModuleList([])
            temp_hidden_size = []
            for v in vision_tower:
                if v is None:
                    self.mm_projector.append(None)
                    temp_hidden_size.append(None)
                else:
                    self.config.mm_hidden_size = v.hidden_size
                    temp_hidden_size.append(v.hidden_size)
                    self.mm_projector.append(build_vision_projector(self.config))
            self.config.mm_hidden_size = temp_hidden_size

        if pretrain_mm_mlp_adapter is not None:
            if "," not in pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            else:
                names_pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter.split(',')
                for n, i in enumerate(names_pretrain_mm_mlp_adapter):
                    mm_projector_weights = torch.load(n, map_location='cpu')
                    def get_w(weights, keyword):
                        return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                    self.mm_projector[i].load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor

def save_tensor_to_folder(tensor, folder_path, max_tensors=100):
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Count the number of tensor files in the folder
    tensor_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    tensor_count = len(tensor_files)

    # Save the current tensor if the count is less than max_tensors
    if tensor_count < max_tensors:
        # Create a unique filename for the tensor
        tensor_filename = os.path.join(folder_path, f'tensor_{tensor_count + 1}.pt')
        torch.save(tensor, tensor_filename)
        print(f'Saved tensor to {tensor_filename}')

    # Exit the program if the number of tensors in the folder reaches max_tensors
    if tensor_count + 1 >= max_tensors:
        print(f'Tensor count has reached {max_tensors}. Exiting the program.')
        exit()


class LlavaMetaForCausalLM(ABC):
    # MOF progress
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        if type(images) is not list:
            image_features = self.get_model().get_vision_tower()(images)
            image_features = self.get_model().mm_projector(image_features)
        else:
            # mof
            # vision_towers = self.get_model().get_vision_tower()
            # mm_projectors = self.get_model().mm_projector
            # image_features_list = []
            # for i in range(len(images)):
            #     if not images[i] is None:
            #         image_features = vision_towers[i](images[i])
            #         image_features_list.append(mm_projectors[i](image_features))
            # if len(image_features_list) == 1:
            #     image_features_list.append(torch.randn_like(image_features_list[0]))
            # image_features = torch.stack(image_features_list, dim=1)
            # B, N, T, D = image_features.shape
            # image_features = image_features.transpose(1, 2).reshape(B, N * T, D)
            vision_tower = self.get_model().get_vision_tower()
            if type(vision_tower) is nn.ModuleList:
                f_list = []
                for i, v in enumerate(vision_tower):
                    image_features = v(images[i])
                    f_list.append(image_features)
                image_features = torch.cat(f_list, dim=-1)
                image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_features(self, images):
        image_features = self.get_model().mm_projector(images)
        return image_features


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(vision_tower) is str:
                image_features = self.encode_features(images)
        else:
            if (type(images) is list or images.ndim == 5) and type(vision_tower) is not nn.ModuleList:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                images = torch.cat([image for image in images], dim=0)
                image_features = self.encode_images(images)
                split_sizes = [image.shape[0] for image in images]
                image_features = torch.split(image_features, split_sizes, dim=0)
                mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
                image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
                if mm_patch_merge_type == 'flat':
                    image_features = [x.flatten(0, 1) for x in image_features]
                elif mm_patch_merge_type.startswith('spatial'):
                    new_image_features = []
                    for image_idx, image_feature in enumerate(image_features):
                        if image_feature.shape[0] > 1:
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if image_aspect_ratio == 'anyres':
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                raise NotImplementedError
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                                ), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = image_feature[0]
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                        new_image_features.append(image_feature)
                    image_features = new_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            else:
                image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # print(cur_image_features.shape)
        # print(self.model.embed_tokens)
        # torch.save(cur_image_features.cpu(), "/mnt/bn/shijiaynas/cat.pt")
        # torch.save(self.model.embed_tokens, "/mnt/bn/shijiaynas/clip_embed_layer.pt")
        # temp_cur_image_features = cur_image_features.view(16, 16, 4096).permute(2, 0, 1)
        # temp_cur_image_features = F.interpolate(temp_cur_image_features.unsqueeze(0).to(torch.float32), size=(24, 24), mode='bilinear', align_corners=False)
        # temp_cur_image_features = temp_cur_image_features.squeeze(0).permute(1, 2, 0).view(-1, 4096)
        # print(temp_cur_image_features.shape)

        # temp_cur_image_features = cur_image_features.view(24, 24, 4096).permute(2, 0, 1)
        # temp_cur_image_features = F.interpolate(temp_cur_image_features.unsqueeze(0).to(torch.float32), size=(16, 16), mode='bilinear', align_corners=False)
        # temp_cur_image_features = temp_cur_image_features.squeeze(0).permute(1, 2, 0).view(-1, 4096)
        # print(temp_cur_image_features.shape)

        # save_tensor_to_folder(cur_image_features.cpu(), '/mnt/bn/shijiaynas/mmbench_tensors3/sd2.1')
        # save_tensor_to_folder(cur_image_features.cpu(), '/mnt/bn/shijiaynas/temp_tensors')

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False