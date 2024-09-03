import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

# The benchmark of which A score you want to compute
base_folder = '/any/path/mmbench'

# The 
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

# Load tensors from the 'clip' subfolder
clip_tensors = load_tensors('clip336')
if not clip_tensors:
    raise ValueError("Failed to load tensors from 'clip' subfolder")

# Dictionary to store the results
results = {}

# Compute the cosine similarity and average results
for subfolder in subfolders:
    if subfolder == 'clip336':
        continue
    
    other_tensors = load_tensors(subfolder)
    if not other_tensors:
        print(f"Skipping {subfolder} due to loading error.")
        continue
    
    cosine_similarities = []
    
    for clip_tensor, other_tensor in tqdm(zip(clip_tensors, other_tensors), total=100, desc=f'Processing {subfolder}'):
        if clip_tensor.shape != other_tensor.shape:
            print(f"Shape mismatch between tensors in 'clip' and '{subfolder}'")
            continue
        
        # Normalize tensors
        clip_tensor = normalize_feat(clip_tensor)
        other_tensor = normalize_feat(other_tensor)
        
        # Reshape tensors for cosine similarity computation
        clip_tensor = clip_tensor.unsqueeze(0)  # shape: [1, 576, 4096]
        other_tensor = other_tensor.unsqueeze(1)  # shape: [576, 1, 4096]
        
        # Compute cosine similarity along the channel dimension
        similarity = F.cosine_similarity(other_tensor, clip_tensor, dim=-1)  # shape: [576, 576]
        
        # Find the max similarity for each vector in other_tensor
        max_similarity = similarity.max(dim=1).values  # shape: [576]
        cosine_similarities.append(max_similarity.mean().item())
    
    # Average across the 100 tensors
    if cosine_similarities:
        avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
        results[subfolder] = avg_cosine_similarity

# Print the results
for subfolder, avg_similarity in results.items():
    print(f'Average cosine similarity between clip and {subfolder}: {avg_similarity}')
