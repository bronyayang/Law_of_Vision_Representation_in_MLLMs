import argparse
import gc
import random
import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from src.models.dift_sd import SDFeaturizer
from src.models.dift_imsd import IMSDFeaturizer
from src.models.dift_dit import DiTFeaturizer
from src.models.dift_sd3 import SD3Featurizer
from src.utils.visualization import Demo
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

import torch
from diffusers import StableDiffusion3Pipeline

# pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# image = pipe(
#     "A cat holding a sign that says hello world",
#     negative_prompt="",
#     num_inference_steps=28,
#     guidance_scale=7.0,
# ).images[0]
# image

# exit()

class args_c:
    def __init__(self):
        self.mm_vision_select_layer = -2

feature = "SD3DIFT"
if feature == "DIFT":
    dift = SDFeaturizer()
elif feature == "IMDIFT":
    dift = IMSDFeaturizer()
elif feature == "DiTDIFT":
    dift = DiTFeaturizer()
elif feature == "SD3DIFT":
    dift = SD3Featurizer()
elif feature == "CLIP":
    args = args_c()
    dift = CLIPVisionTower(vision_tower='openai/clip-vit-large-patch14', args=args)

# you can choose visualize cat or guitar
# category = random.choice(['cat', 'guitar'])
category ='cat'

print(f"let's visualize semantic correspondence on {category}")

if category == 'cat':
    filelist = ['/opt/tiger/LLaVA1.5/llava/eval/scat/orange_cat.jpg', '/opt/tiger/LLaVA1.5/llava/eval/scat/shelf_cat.png']
elif category == 'guitar':
    filelist = ['./assets/guitar.png', './assets/target_guitar.png']
elif category == '0':
    filelist = ['/viscam/projects/part_decomp/dift/eval_images/0/GCC_train_000261240.jpg', '/viscam/projects/part_decomp/dift/eval_images/0/GCC_train_002127756.jpg']
elif category == '1':
    filelist = ['/viscam/projects/part_decomp/dift/eval_images/1/GCC_train_000131600.jpg', '/viscam/projects/part_decomp/dift/eval_images/1/GCC_train_000156948.jpg']
elif category == '2':
    filelist = ['/viscam/projects/part_decomp/dift/eval_images/2/GCC_train_000691494.jpg', '/viscam/projects/part_decomp/dift/eval_images/2/GCC_train_002280061.jpg']
elif category == '3':
    filelist = ['/viscam/projects/part_decomp/dift/eval_images/3/GCC_train_000235136.jpg', '/viscam/projects/part_decomp/dift/eval_images/3/GCC_train_002290062.jpg']
elif category == '4':
    filelist == ['/viscam/projects/part_decomp/dift/eval_images/4/GCC_train_000221071.jpg', '/viscam/projects/part_decomp/dift/eval_images/4/GCC_train_000324898.jpg']
elif category == '5':
    filelist = ['/viscam/projects/part_decomp/dift/eval_images/5/GCC_train_002329580.jpg', '/viscam/projects/part_decomp/dift/eval_images/5/GCC_train_002514353.jpg']
elif category == '6':
    filelist = ['/viscam/projects/part_decomp/dift/eval_images/6/GCC_train_002501270.jpg', '/viscam/projects/part_decomp/dift/eval_images/6/GCC_train_002534171.jpg']
elif category == '7':
    filelist = ['/viscam/projects/part_decomp/dift/eval_images/7/GCC_train_001915506.jpg', '/viscam/projects/part_decomp/dift/eval_images/7/GCC_train_002538120.jpg']
elif category == '8':
    filelist = ['/viscam/projects/part_decomp/dift/eval_images/8/GCC_train_001163466.jpg', '/viscam/projects/part_decomp/dift/eval_images/8/GCC_train_001164170.jpg']
elif category == '9':
    filelist = ['/viscam/projects/part_decomp/dift/eval_images/9/GCC_train_000141221.jpg', '/viscam/projects/part_decomp/dift/eval_images/9/GCC_train_002462785.jpg']



# prompt = f'a photo of a {category}'
prompt = ''
ensemble_size = 1

ft = []
imglist = []

# decrease these two if you don't have enough RAM or GPU memory
if feature == "DIFT":
    img_size = 768
elif feature == "IMDIFT":
    img_size = 768
elif feature == "CLIP":
    img_size = 224
elif feature == "DiTDIFT":
    img_size = 512
elif feature == "DINOv2":
    img_size = 518
elif feature == 'SD3DIFT':
    img_size = 1024
ensemble_size = 1

for filename in filelist:
    img = Image.open(filename).convert('RGB')
    img = img.resize((img_size, img_size))
    imglist.append(img)
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
    if feature == "DIFT":
        ft.append(dift.forward(img_tensor,
                            prompt=prompt,
                            ensemble_size=ensemble_size))
    elif feature == "IMDIFT":
        f = dift.forward(img_tensor=img_tensor.unsqueeze(0),
                            prompt=prompt,
                            ensemble_size=ensemble_size)
        print(f.unsqueeze(0).shape)
        ft.append(f.unsqueeze(0))
    elif feature == "CLIP":
        f = dift.forward(img_tensor.unsqueeze(0))
        print(f.shape)

    elif feature == "DINOv2":
        f = dift.forward(img_tensor.unsqueeze(0))
        print(f.shape)
        # (1, 256, 1024)
        f = f[:, 1:, :]
        ft.append(f.permute(0,2,1).view(1, 1024, 37, 37))
    elif feature == "DiTDIFT":
        f = dift.forward(img_tensor.unsqueeze(0).to(torch.bfloat16),
                        prompt=prompt,
                        ensemble_size=ensemble_size)
        print(f.shape)
        # (1, 256, 1024)
        ft.append(f)
    elif feature == "SD3DIFT":
        f = dift.forward(img_tensor.unsqueeze(0).to(torch.bfloat16),
                        prompt=prompt,
                        ensemble_size=ensemble_size)
        print(f.shape)
        # (1, 256, 1024)
        ft.append(f)

ft = torch.cat(ft, dim=0)
print(ft.shape)
gc.collect()
torch.cuda.empty_cache()