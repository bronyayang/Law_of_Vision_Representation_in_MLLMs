import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

# The benchmark of which A score you want to compute
base_folder = '/any/path/mmbench'

# The subfolders for features
subfolders = ['clip336', 'clip224', 'dino', 'dit', 'imsd', 'openclip', 'sd1.5', 'sd2.1', 'sd3', 'sdxl']

def normalize_feat(feat, epsilon=1e-10):
    norms = torch.linalg.norm(feat, dim=-1, keepdim=True)
    norm_feats = feat / (norms + epsilon)
    return norm_feats

# Function to load tensors from a subfolder
def load_tensors(subfolder):
    tensors = []
    for i in range(1, 101):
        tensor_path = os.path.join(base_folder, subfolder, f"tensor_{i}.pt")
        try:
            tensor = torch.load(tensor_path)
            tensors.append(tensor)
        except Exception as e:
            print(f"Error loading {tensor_path}: {e}")
            return []
    return tensors

# Load tensors from 'clip336' and 'clip224' subfolders
clip336_tensors = load_tensors('clip336')
clip224_tensors = load_tensors('clip224')

if not clip336_tensors or not clip224_tensors:
    raise ValueError("Failed to load tensors from 'clip336' or 'clip224' subfolder")

# Dictionary to store the results
results = {}

# Compute the cosine similarity and average results for each subfolder
for subfolder in subfolders:
    
    other_tensors = load_tensors(subfolder)
    if not other_tensors:
        print(f"Skipping {subfolder} due to loading error.")
        continue
    
    cosine_similarities_336 = []
    cosine_similarities_224 = []
    
    for clip336_tensor, clip224_tensor, other_tensor in tqdm(zip(clip336_tensors, clip224_tensors, other_tensors), total=100, desc=f'Processing {subfolder}'):
        
        # Normalize tensors
        clip336_tensor = normalize_feat(clip336_tensor)
        clip224_tensor = normalize_feat(clip224_tensor)
        other_tensor = normalize_feat(other_tensor)
        
        # Reshape tensors for cosine similarity computation
        clip336_tensor = clip336_tensor.unsqueeze(0)  # shape: [1, x, 4096]
        clip224_tensor = clip224_tensor.unsqueeze(0)  # shape: [1, x, 4096]
        other_tensor = other_tensor.unsqueeze(1)  # shape: [x, 1, 4096]
        
        # Compute cosine similarity along the channel dimension
        similarity_336 = F.cosine_similarity(other_tensor, clip336_tensor, dim=-1)  # shape: [x, x]
        similarity_224 = F.cosine_similarity(other_tensor, clip224_tensor, dim=-1)  # shape: [x, x]
        
        # Find the max similarity for each vector in other_tensor
        max_similarity_336 = similarity_336.max(dim=1).values  # shape: [x]
        max_similarity_224 = similarity_224.max(dim=1).values  # shape: [x]
        
        cosine_similarities_336.append(max_similarity_336.mean().item())
        cosine_similarities_224.append(max_similarity_224.mean().item())
    
    # Average across the 100 tensors
    if cosine_similarities_336 and cosine_similarities_224:
        avg_cosine_similarity_336 = sum(cosine_similarities_336) / len(cosine_similarities_336)
        avg_cosine_similarity_224 = sum(cosine_similarities_224) / len(cosine_similarities_224)
        
        # Take the average of similarities from clip336 and clip224
        avg_cosine_similarity = (avg_cosine_similarity_336 + avg_cosine_similarity_224) / 2
        results[subfolder] = avg_cosine_similarity

# Print the results
for subfolder, avg_similarity in results.items():
    print(f'Average cosine similarity between clip224+clip336 and {subfolder}: {avg_similarity}')
