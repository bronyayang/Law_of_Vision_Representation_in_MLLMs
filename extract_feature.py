import os
import torch
from PIL import Image
import torch
from tqdm import tqdm
from torchvision.transforms import PILToTensor
from llava.model.multimodal_encoder.diffLVLM.src.models.dift_sd import SDFeaturizer
from llava.model.multimodal_encoder.diffLVLM.src.models.dift_imsd import IMSDFeaturizer
from llava.model.multimodal_encoder.diffLVLM.src.models.dift_dit import DiTFeaturizer
from llava.model.multimodal_encoder.diffLVLM.src.models.dift_sd3 import SD3Featurizer
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from llava.model.multimodal_encoder.dinov2_encoder import DinoV2VisionTower
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower

# Define the input and output paths
input_path = '/opt/tiger/LLaVA1.5/GeoAware-SC/data/SPair-71k/JPEGImages'
output_path = '/mnt/bn/shijiaynas/spair_feature/dino336'

class args_c:
    def __init__(self):
        self.mm_vision_select_layer = -2

feature = "DINOv2"
if feature == "DIFT2.1":
    dift = SDFeaturizer(sd_id='stabilityai/stable-diffusion-2-1')
if feature == "DIFT1.5":
    dift = SDFeaturizer(sd_id='runwayml/stable-diffusion-v1-5')
elif feature == "DIFTXL":
    dift = SDFeaturizer(sd_id='stabilityai/stable-diffusion-xl-base-1.0')
elif feature == "IMDIFT":
    dift = IMSDFeaturizer()
elif feature == "DiTDIFT":
    dift = DiTFeaturizer()
elif feature == "SD3DIFT":
    dift = SD3Featurizer()
elif feature == "CLIP":
    args = args_c()
    dift = CLIPVisionTower(vision_tower='openai/clip-vit-large-patch14', args=args)
elif feature == "OPENCLIP":
    args = args_c()
    dift = CLIPVisionTower(vision_tower='laion/CLIP-ViT-L-14-laion2B-s32B-b82K', args=args)
elif feature == "DINOv2":
    args = args_c()
    dift = DinoV2VisionTower(vision_tower='facebook/dinov2-large', args=args)
elif feature == "SigLIP":
    args = args_c()
    dift = SigLipVisionTower(vision_tower='google/siglip-base-patch16-224', args=args).to(torch.bfloat16)

dift.eval()
dift.cuda()


# Function to extract features from an image
def extract_features(image_path):
    if feature == "DIFT2.1" or feature == "DIFT1.5" or feature == "IMDIFT":
        img_size = 768
    elif feature == "CLIP" or feature == "OPENCLIP":
        img_size = 224
    elif feature == "DiTDIFT" or feature == 'SD3DIFT' or feature == "DIFTXL":
        img_size = 512
    elif feature == "DINOv2" or feature == "SigLIP":
        img_size = 224
    ensemble_size = 1
    prompt = ''
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
    if feature == "DIFT1.5" or feature == "DIFT2.1" or feature == "DIFTXL":
        f = dift.forward(img_tensor.unsqueeze(0).to(torch.bfloat16),
                    prompt=prompt,
                    ensemble_size=ensemble_size)
        print(f.unsqueeze(0).shape)
        return f.unsqueeze(0)
    elif feature == "IMDIFT":
        f = dift.forward(img_tensor=img_tensor.unsqueeze(0).to(torch.bfloat16),
                            prompt=prompt,
                            ensemble_size=ensemble_size)
        print(f.unsqueeze(0).shape)
        return f.unsqueeze(0)
    elif feature == "CLIP" or feature == "OPENCLIP":
        f = dift.forward(img_tensor.unsqueeze(0))
        return f.permute(0,2,1).view(1, 1024, 16, 16)
    elif feature == "DINOv2":
        f = dift.forward(img_tensor.unsqueeze(0))
        print(f.shape)
        f = f.permute(0,2,1).view(1, 1024, 24, 24)
        return f
    elif feature == "SigLIP":
        f = dift.forward(img_tensor.unsqueeze(0).to(torch.float32))
        f = f.permute(0,2,1).view(1, 768, 14, 14)
        return f
    elif feature == "DiTDIFT":
        f = dift.forward(img_tensor.unsqueeze(0).to(torch.bfloat16),
                        prompt=prompt,
                        ensemble_size=ensemble_size)
        f = f.permute(0, 1, 3,2).view(1, 4608, 16, 16)
        print(f.shape)
        # (1, 256, 1024)
        return f
    elif feature == "SD3DIFT":
        f = dift.forward(img_tensor.unsqueeze(0).to(torch.bfloat16),
                        prompt=prompt,
                        ensemble_size=ensemble_size)
        print(f.shape)
        # (1, 256, 1024)
        return f


# Function to process all images in the directory
def process_images(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # Get the class and image name from the input path
                class_name = os.path.basename(root)
                image_name = os.path.splitext(file)[0]

                # Construct the full input path
                input_image_path = os.path.join(root, file)

                # Extract features from the image
                features = extract_features(input_image_path)

                # Construct the full output path and create directories if needed
                output_image_path = os.path.join(output_dir, class_name, f'{image_name}_dino336.pt')
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                # Save the features as a numpy file
                torch.save(features, output_image_path)
                print(f'Saved features to {output_image_path}')

# Process all images
process_images(input_path, output_path)
