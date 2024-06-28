import transformers
import torch
import os
import copy
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
from llava.model.multimodal_encoder.builder import build_vision_tower, build_diffusion_vision_tower, build_dinov2_vision_tower
from llava.train.train import ModelArguments, TrainingArguments, DataArguments, LazySupervisedDataset, \
    DataCollatorForSupervisedDataset, preprocess_multimodal, preprocess
from typing import Dict, Optional, Sequence, List
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from llava.constants import IGNORE_INDEX
from PIL import Image

build_function_mapping = {'openai/clip-vit-large-patch14-336': build_vision_tower, 
                        'stabilityai/stable-diffusion-2-1': build_diffusion_vision_tower,
                        'stabilityai/stable-diffusion-1-5': build_diffusion_vision_tower,
                        'runwayml/stable-diffusion-v1-5': build_diffusion_vision_tower,
                        'lambdalabs/sd-image-variations-diffusers': build_diffusion_vision_tower,
                        'facebook/dinov2-large': build_dinov2_vision_tower,
                        'stabilityai/stable-diffusion-xl-base-1.0': build_diffusion_vision_tower}

os_path={
        "coco/train2017": "/mnt/bn/bohanzhainas1/Public_data/mscoco_karpathy/train2017",
        "sam/images": "/mnt/bn/bohanzhainas1/yiqi.wang/data/sharegpt4v/sam/images",
        "share_textvqa/images": "/mnt/bn/bohanzhainas1/yiqi.wang/data/sharegpt4v/data/share_textvqa/images",
        'web-celebrity/images': "/mnt/bn/bohanzhainas1/yiqi.wang/data/sharegpt4v/data/web-celebrity/images",
        'web-landmark/images':"/mnt/bn/bohanzhainas1/yiqi.wang/data/sharegpt4v/data/web-landmark/images",
        'wikiart/images':"/mnt/bn/bohanzhainas1/yiqi.wang/data/sharegpt4v/data/wikiart/images",
        "vg/VG_100K": "/mnt/bn/bohanzhainas1/haogeng/llava_images/vg/VG_100K",
        'vg/VG_100K_2': "/mnt/bn/bohanzhainas1/haogeng/llava_images/vg/VG_100K_2",
        'gqa/images':"/mnt/bn/bohanzhainas1/haogeng/llava_images/gqa/images",
        'ocr_vqa/images':"/mnt/bn/bohanzhainas1/haogeng/llava_images/ocr_vqa/images",
        "textvqa/train_images": "/mnt/bn/bohanzhainas1/haogeng/llava_images/textvqa/train_images",
        "llava/llava_pretrain/images": "/mnt/bn/bohanzhainas1/Public_data/llava_1.3_data/LLaVA-Pretrain/"

    }


match_path={
        "mscoco_karpathy/train2017": "/mnt/bn/shijiaynas/LLaVA_finetune_sd1.5/coco/train2017",
        "sam/images": "/mnt/bn/shijiaynas/LLaVA_finetune_sd1.5/sam/images",
        "share_textvqa/images": "/mnt/bn/shijiaynas/LLaVA_finetune_sd1.5/share_textvqa/images",
        'web-celebrity/images': "/mnt/bn/shijiaynas/LLaVA_finetune_sd1.5/web-celebrity/images",
        'web-landmark/images':"/mnt/bn/shijiaynas/LLaVA_finetune_sd1.5/web-landmark/images",
        'wikiart/images':"/mnt/bn/shijiaynas/LLaVA_finetune_sd1.5/wikiart/images",
        "vg/VG_100K": "/mnt/bn/shijiaynas/LLaVA_finetune_sd1.5/vg/VG_100K",
        'vg/VG_100K_2': "/mnt/bn/shijiaynas/LLaVA_finetune_sd1.5/vg/VG_100K_2",
        'gqa/images':"/mnt/bn/shijiaynas/LLaVA_finetune_sd1.5/gqa/images",
        'ocr_vqa/images':"/mnt/bn/shijiaynas/LLaVA_finetune_sd1.5/ocr_vqa/images",
        "textvqa/train_images": "/mnt/bn/shijiaynas/LLaVA_finetune_sd1.5/textvqa/train_images",
        "LLaVA-Pretrain": "/mnt/bn/shijiaynas/LLaVA_pretrain_sd1.5"

    }

def cleanup():
    dist.destroy_process_group()

class ModDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        batch['path'] = [i['path'] if 'path' in i.keys() else None for i in instances]
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if type(images[0]) is list:
                batch['images'] = []
                for i in range(len(images[0])):
                    if all(x[i] is not None and x[i].shape == images[0][i].shape for x in images):
                        batch['images'].append(torch.stack([x[i] for x in images]))
                    elif all(x[i] is None for x in images):
                        batch['images'].append(None)
                    else:
                        batch['images'] = images
            else:
                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch['images'] = torch.stack(images)
                else:
                    batch['images'] = images
        return batch

class ModLazySupervisedDataset(LazySupervisedDataset):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            # image_folder = self.data_args.image_folder
            if self.data_args.image_folder is None:
                for img_folder in os_path.keys():
                    img_folder_tmp = img_folder + '/'
                    if img_folder_tmp in image_file:
                        image_file = image_file.replace(img_folder_tmp, '')
                        image_folder = os_path[img_folder]
                        break
            else:
                image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            path = os.path.join(image_folder, image_file)
            image = Image.open(path).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                if type(processor) is list:
                    image_list = []
                    for p in processor:
                        if p is None:
                            image_list.append(None)
                        else:
                            image_list.append(p.preprocess(image, return_tensors='pt')['pixel_values'][0])
                    image = image_list
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            data_dict['path'] = path
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            processor = self.data_args.image_processor
            if type(processor) is list:
                image_list = []
                for p in processor:
                    if p is None:
                        image_list.append(None)
                    else:
                        crop_size = p.crop_size
                        image_list.append(torch.zeros(3, crop_size['height'], crop_size['width']))
                data_dict['image'] = image_list
            else:
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

def make_mod_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ModLazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = ModDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def inference(model_args, data_args, training_args):
    # create model
    model = build_function_mapping[model_args.vision_tower](model_args)
    data_args.image_processor = model.image_processor
    model.dummy_param = nn.Parameter(torch.empty(0), requires_grad=True)
    model = model.to(dist.get_rank())

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
    
    # Prepare DataLoader
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    data_module = make_mod_supervised_data_module(tokenizer=tokenizer,
                                        data_args=data_args)

    sampler = DistributedSampler(data_module['train_dataset'], num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    dataloader = DataLoader(data_module['train_dataset'], batch_size=training_args.per_device_train_batch_size, sampler=sampler, collate_fn=ModDataCollatorForSupervisedDataset(tokenizer=tokenizer))
    model.eval()

    # pretrain extract
    with torch.no_grad():
        for data_dict in tqdm(dataloader):
            if 'path' in data_dict.keys():
                path = data_dict['path']
                if path[0] is not None:
                    path_l = path[0].split('/')
                    file_name = path_l[-1].split('.')[0]
                    if not os.path.exists(f'/mnt/bn/shijiaynas/obelisc/{path_l[-2]}/{file_name}.pt'):
                        if not os.path.exists(f'/mnt/bn/shijiaynas/obelisc/{path_l[-2]}'):
                            os.mkdir(f'/mnt/bn/shijiaynas/obelisc/{path_l[-2]}')
                        images = data_dict['images'].to(dtype=torch.bfloat16)
                        outputs = model(images)
                        out_list = torch.split(outputs, split_size_or_sections=1)
                        torch.save(out_list[0].squeeze().cpu(), f'/mnt/bn/shijiaynas/obelisc/{path_l[-2]}/{file_name}.pt')

    # # finetune extract
    # with torch.no_grad():
    #     for data_dict in tqdm(dataloader):
    #             path = data_dict['path']
    #             if path[0] is not None:
    #                 path_l = path[0].split('/')
    #                 file_name = path_l[-1].split('.')[0]
    #                 for k in match_path.keys():
    #                     if k in path[0]:
    #                         # if not os.path.exists(f'{match_path[k]}/{file_name}.pt'):
    #                         images = data_dict['images'].to(dtype=torch.bfloat16)
    #                         outputs = model(images)
    #                         out_list = torch.split(outputs, split_size_or_sections=1)
    #                         torch.save(out_list[0].squeeze().cpu(), f'{match_path[k]}/{file_name}.pt')


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # dist.init_process_group("nccl")

    inference(model_args, data_args, training_args)

    # Synchronize all processes to ensure the saving is completed in the master process
    dist.barrier()
    cleanup()