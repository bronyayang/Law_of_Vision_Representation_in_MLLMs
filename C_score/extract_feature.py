import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import PILToTensor
from llava.model.multimodal_encoder.diffLVLM.src.models.dift_sd import SDFeaturizer
from llava.model.multimodal_encoder.diffLVLM.src.models.dift_imsd import IMSDFeaturizer
from llava.model.multimodal_encoder.diffLVLM.src.models.dift_dit import DiTFeaturizer
from llava.model.multimodal_encoder.diffLVLM.src.models.dift_sd3 import SD3Featurizer
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from llava.model.multimodal_encoder.dinov2_encoder import DinoV2VisionTower
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower


class args_c:
    def __init__(self):
        self.mm_vision_select_layer = -2


def parse_args():
    parser = argparse.ArgumentParser(description='Extract vision features from images')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input images directory (e.g., ./data/SPair-71k/JPEGImages)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save extracted features (e.g., ./data/SPair-71k/features)')
    parser.add_argument('--feature', type=str, required=True,
                        choices=['DIFT1.5', 'DIFT2.1', 'DIFTXL', 'IMDIFT', 'DiTDIFT', 'SD3DIFT',
                                 'CLIP', 'OPENCLIP', 'DINOv2', 'SigLIP'],
                        help='Vision representation to extract')
    return parser.parse_args()


def initialize_feature_extractor(feature_type):
    """Initialize the feature extractor based on the specified type."""
    if feature_type == "DIFT2.1":
        dift = SDFeaturizer(sd_id='stabilityai/stable-diffusion-2-1')
    elif feature_type == "DIFT1.5":
        dift = SDFeaturizer(sd_id='runwayml/stable-diffusion-v1-5')
    elif feature_type == "DIFTXL":
        dift = SDFeaturizer(sd_id='stabilityai/stable-diffusion-xl-base-1.0')
    elif feature_type == "IMDIFT":
        dift = IMSDFeaturizer()
    elif feature_type == "DiTDIFT":
        dift = DiTFeaturizer()
    elif feature_type == "SD3DIFT":
        dift = SD3Featurizer()
    elif feature_type == "CLIP":
        args = args_c()
        dift = CLIPVisionTower(vision_tower='openai/clip-vit-large-patch14', args=args)
    elif feature_type == "OPENCLIP":
        args = args_c()
        dift = CLIPVisionTower(vision_tower='laion/CLIP-ViT-L-14-laion2B-s32B-b82K', args=args)
    elif feature_type == "DINOv2":
        args = args_c()
        dift = DinoV2VisionTower(vision_tower='facebook/dinov2-large', args=args)
    elif feature_type == "SigLIP":
        args = args_c()
        dift = SigLipVisionTower(vision_tower='google/siglip-base-patch16-224', args=args).to(torch.bfloat16)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    dift.eval()
    dift.cuda()
    return dift


def extract_features(image_path, feature_type, feature_extractor):
    """Extract features from an image using the specified feature extractor."""
    if feature_type in ["DIFT2.1", "DIFT1.5", "IMDIFT"]:
        img_size = 768
    elif feature_type in ["CLIP", "OPENCLIP"]:
        img_size = 224
    elif feature_type in ["DiTDIFT", "SD3DIFT", "DIFTXL"]:
        img_size = 512
    elif feature_type in ["DINOv2", "SigLIP"]:
        img_size = 224
    else:
        img_size = 224  # default
    
    ensemble_size = 1
    prompt = ''
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
    
    if feature_type in ["DIFT1.5", "DIFT2.1", "DIFTXL"]:
        f = feature_extractor.forward(img_tensor.unsqueeze(0).to(torch.bfloat16),
                                      prompt=prompt,
                                      ensemble_size=ensemble_size)
        print(f.unsqueeze(0).shape)
        return f.unsqueeze(0)
    elif feature_type == "IMDIFT":
        f = feature_extractor.forward(img_tensor=img_tensor.unsqueeze(0).to(torch.bfloat16),
                                      prompt=prompt,
                                      ensemble_size=ensemble_size)
        print(f.unsqueeze(0).shape)
        return f.unsqueeze(0)
    elif feature_type in ["CLIP", "OPENCLIP"]:
        f = feature_extractor.forward(img_tensor.unsqueeze(0))
        return f.permute(0, 2, 1).view(1, 1024, 16, 16)
    elif feature_type == "DINOv2":
        f = feature_extractor.forward(img_tensor.unsqueeze(0))
        print(f.shape)
        f = f.permute(0, 2, 1).view(1, 1024, 24, 24)
        return f
    elif feature_type == "SigLIP":
        f = feature_extractor.forward(img_tensor.unsqueeze(0).to(torch.float32))
        f = f.permute(0, 2, 1).view(1, 768, 14, 14)
        return f
    elif feature_type == "DiTDIFT":
        f = feature_extractor.forward(img_tensor.unsqueeze(0).to(torch.bfloat16),
                                      prompt=prompt,
                                      ensemble_size=ensemble_size)
        f = f.permute(0, 1, 3, 2).view(1, 4608, 16, 16)
        print(f.shape)
        return f
    elif feature_type == "SD3DIFT":
        f = feature_extractor.forward(img_tensor.unsqueeze(0).to(torch.bfloat16),
                                      prompt=prompt,
                                      ensemble_size=ensemble_size)
        print(f.shape)
        return f
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def process_images(input_dir, output_dir, feature_type, feature_extractor):
    """Process all images in the directory and extract features."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # Get the class and image name from the input path
                class_name = os.path.basename(root)
                image_name = os.path.splitext(file)[0]

                # Construct the full input path
                input_image_path = os.path.join(root, file)

                # Extract features from the image
                features = extract_features(input_image_path, feature_type, feature_extractor)

                # Construct the full output path and create directories if needed
                output_image_path = os.path.join(output_dir, class_name, f'{image_name}.pt')
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                # Save the features
                torch.save(features, output_image_path)
                print(f'Saved features to {output_image_path}')


def main():
    # Parse command-line arguments
    args = parse_args()
    
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Feature type: {args.feature}")
    
    # Initialize feature extractor
    print(f"Initializing {args.feature} feature extractor...")
    feature_extractor = initialize_feature_extractor(args.feature)
    
    # Process all images
    print("Processing images...")
    process_images(args.input_path, args.output_path, args.feature, feature_extractor)
    print("Done!")


if __name__ == "__main__":
    main()
