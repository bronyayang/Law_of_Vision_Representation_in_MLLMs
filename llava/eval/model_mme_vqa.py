import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math




eval_type_dict = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
}
image_path_map = {
    'posters': 'posters/images',
    'celebrity': 'celebrity/images',
    'scene': 'scene/images',
    'landmark': 'landmark/images',
    'artwork': 'artwork/images'
}

def load_questions(source_path): 
    lines = open(source_path, 'r').readlines()
    question_ids = []
    questions = []
    img_names = []
    gt_anses = []
    for id, line in enumerate(lines):
        img_name, question, gt_ans, pred_ans = line.split("\t")
        questions.append(question)
        img_names.append(img_name)
        gt_anses.append(gt_ans)
        question_ids.append(id)
    
    return questions, img_names, gt_anses, question_ids


def load_mme_question(question_folder, tasks):
    rt_obj = {}
    for task_name in tasks:
        task_txt = os.path.join(question_folder, task_name + ".txt")
        questions, img_names, gt_anses, question_ids = load_questions(task_txt)
        rt_obj[task_name] = {'questions':questions, 'img_names':img_names, 'gt_anses':gt_anses, 'question_ids':question_ids}
    return rt_obj


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    question_collections = load_mme_question(args.question_folder, eval_type_dict['Perception'])
    # answers_folder = os.path.expanduser(args.answers_folder)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # ans_file = open(answers_file, "w")
    for task in question_collections.keys():
        questions = question_collections[task]['questions']
        imgs = question_collections[task]['img_names']
        gt_anses = question_collections[task]['gt_anses']
        answers_file = os.path.join(args.answers_folder, task+'.txt')
        task_lines = []
        for i, line in tqdm(enumerate(questions)):
            image_file = imgs[i]
            qs = line
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            if task in image_path_map:
                task = image_path_map[task]
            image_path = os.path.join(args.image_folder, task, image_file)
            if not os.path.exists(image_path):
                continue
            image = Image.open(image_path)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            model.cuda()
            input_ids.cuda()
            image_tensor.cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
            
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            new_rt = [imgs[i], line, gt_anses[i], outputs]
            new_line = '\t'.join(new_rt) + '\n'
            task_lines.append(new_line)
        with open(answers_file, 'w') as file:
            file.writelines(task_lines)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question_folder", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers_folder", type=str, default="/opt/tiger/LLaVA1.5/Awesome-Multimodal-Large-Language-Models/tools/eval_tool/MME_results_llava")
    parser.add_argument("--conv-mode", type=str, default="v1")
    # parser.add_argument("--num-chunks", type=int, default=1)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    # parser.add_argument('--results_dir', default='./LaVIN', type=str)
    args = parser.parse_args()
    eval_model(args)
