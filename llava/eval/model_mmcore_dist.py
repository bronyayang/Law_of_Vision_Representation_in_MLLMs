import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from functools import partial
from typing import Tuple, Dict, Any, Type

from typing import Dict, Optional, Sequence, List

import torch

import transformers
from dataclasses import dataclass, field

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from llava.train.llava_trainer import LLaVATrainer, TrainerForMMLLM
from llava.train.train import DataCollatorForSupervisedDataset
from transformers.trainer import DataCollator
from llava.model import *

from transformers import GenerationConfig, AutoTokenizer


from PIL import Image
import math
import os

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    data_type: str = 'random'

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)
    special_prompt: str = ""
    output_file: str=""


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def concat_images(img1, img2, spacing=10):    
    # Get dimensions of the images
    width1, height1 = img1.size
    width2, height2 = img2.size
    
    # Create a white image with the same height as the images and the width of the spacing
    white_space = Image.new('RGB', (spacing, max(height1, height2)), 'white')
    
    # Create a new image with width = width1 + width2 + spacing, and height = max(height1, height2)
    new_img = Image.new('RGB', (width1 + width2 + spacing, max(height1, height2)))
    
    # Paste the images and the white space on the new image
    new_img.paste(img1, (0, 0))
    new_img.paste(white_space, (width1, 0))
    new_img.paste(img2, (width1 + spacing, 0))
    
    return new_img

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, tokenizer, image_processor, model_config, mm_vet=None, special_prompt=None):
        self.questions = questions
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.mm_vet_path = mm_vet

    def __getitem__(self, index):
        line = self.questions[index]
        qs = line["Question"] + " let's think step by step." # + "Please also provide Reasoning Step, like 1., 2. 3. etc."
        # qs = ''
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs


        conv = conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt)
        # exit()

        image_paths = line["Image path"]
        image_list = []
        for image_path in image_paths:
            image_file = os.path.join(self.mm_vet_path, image_path)
            image = Image.open(image_file).convert('RGB')
            image_list.append(image)
        image_cat = image_list[0]
        for i in range(1, len(image_list)):
            image_cat = concat_images(image_cat, image_list[0])
        
        image_tensor = process_images(image_list, self.image_processor, self.model_config)

        # image_tensor = process_images([image_cat], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return {'input_ids':input_ids, 'image':image_tensor}

    def __len__(self):
        return len(self.questions)

@dataclass
class DataCollatorForInfernceDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance['input_ids'] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids
        )
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, special_prompt="")
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader



def eval_model():
    # global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, testing_args = parser.parse_args_into_dataclasses()
    generation_config = GenerationConfig.from_pretrained(model_args.model_name_or_path)
    # generation_config = GenerationConfig.from_pretrained('/mnt/bn/yukunfeng-nasdrive/bohan/weights/LLaVA_weights/ORI_CKPTs/llava-v1.5-7b')
    generation_config.max_new_tokens = 256
    generation_config.do_sample=True
    generation_config.temperature = 1
    testing_args.generation_config = generation_config
    global local_rank

    local_rank = testing_args.local_rank

    data_path = data_args.data_path # '/opt/tiger/LLaVA1.5/MM-Vet/mm-vet/mm-vet.json'
    data_dict = json.load(open(data_path))
    data = [value for key, value in sorted(data_dict.items(), key=lambda item: int(item[0]))]
    data_keys = [key for key, value in sorted(data_dict.items(), key=lambda item: int(item[0]))]
    # data = data[:1]
    # data_keys = data_keys[:1]
    # print(data)
    disable_torch_init()
    model_path = os.path.expanduser(model_args.model_name_or_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    # Load model, tokenizer and image_processor
    # model = LlavaLlamaForCausalLM.from_pretrained(
    #             model_args.model_name_or_path,
    #             cache_dir=testing_args.cache_dir,
    #         )
    # model.config.use_cache = True
    # model.model.requires_grad_(False)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         model_args.model_name_or_path,
    #     )
    # model.get_model().initialize_vision_modules(
    #         model_args=model_args,
    #         fsdp=testing_args.fsdp
    #     )
    
    # mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    # mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    # if mm_use_im_patch_token:
    #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    # if mm_use_im_start_end:
    #     tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    # model.resize_token_embeddings(len(tokenizer))
    # vision_tower = model.get_vision_tower()
    # if not vision_tower.is_loaded:
    #     vision_tower.load_model()
    # vision_tower = model.get_vision_tower()
    # vision_tower.to(dtype=torch.bfloat16 if testing_args.bf16 else torch.float16)
    # image_processor = vision_tower.image_processor
    

    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    # test_args = TrainingArguments(
    #     output_dir = args.output_folder,
    #     do_train = False,
    #     do_predict = True,
    #     per_device_eval_batch_size = 1,   
    #     dataloader_drop_last = False    
    # )
    data_collator = DataCollatorForInfernceDataset(tokenizer=tokenizer)
    trainer = TrainerForMMLLM(
              model = model,
              tokenizer = tokenizer,
              args = testing_args,
              train_dataset=None,
              eval_dataset=None,
              train_collator=data_collator,
              eval_collator=data_collator)

    # data_module = make_supervised_data_module(tokenizer=tokenizer,
    #                                           data_args=data_args)
    # trainer = LLaVATrainer(model=model,
    #                 tokenizer=tokenizer,
    #                 args=training_args,
    #                 **data_module)
    
    # questions = [json.loads(line) for line in open(args.question_file)]
    # questions = {question['question_id']: question for question in questions}
    
    # # data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    # '/opt/tiger/LLaVA1.5/MM-Vet/mm-vet/images'
    test_dataset = CustomDataset(data, tokenizer, image_processor, model.config, mm_vet=data_args.image_folder, special_prompt=testing_args.special_prompt)

    test_results = trainer.predict(test_dataset)
    output_json = {}
    if local_rank == 0:
        for i, test_result in enumerate(test_results.predictions):
            outputs = tokenizer.batch_decode([test_result], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            key = data_keys[i]
            output_json[key] = outputs

    # with open('/opt/tiger/LLaVA1.5/outputs/llava1.5_llama2_7b_chat.json', 'w') as f:
    #     json.dump(output_json, f, indent=4)
    with open(testing_args.output_file, 'w') as f:
        json.dump(output_json, f, indent=4)

    # print(type(test_results))
    # print(test_results.predictions.shape)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--data_type", type=str, default=None)
    # parser.add_argument("--output_folder", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, default="/mnt/bn/data-tns-algo-masp/data/coco/val2017")
    # parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    # parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    # parser.add_argument("--conv-mode", type=str, default="llava_v1")
    # parser.add_argument("--num-chunks", type=int, default=1)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--temperature", type=float, default=0.2)
    # parser.add_argument("--top_p", type=float, default=None)
    # parser.add_argument("--num_beams", type=int, default=1)
    # args = parser.parse_args()

    eval_model()