import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Define the path to the folder
base_folder = '/mnt/bn/shijiaynas/seedbench_tensors'

# Subfolders
# subfolders = ['clip', 'clip224', 'openclip', 'dinov2', 'imsd', 'sd1.5', 'sdxl','dit', 'sd3', 'sd2.1', 'siglip']
subfolders = ['clipdino', 'clipdino336']

def normalize_feat(feat, epsilon=1e-10):
    norms = torch.linalg.norm(feat, dim=-1)[:, None]
    norm_feats = feat / (norms + epsilon)
    # norm_feats = feats / norms
    
    return norm_feats

# Function to load tensors from a subfolder
def load_tensors(subfolder):
    tensors = []
    for i in range(1, 101):
        tensor_path = os.path.join(base_folder, subfolder, f'tensor_{i}.pt')
        tensor = torch.load(tensor_path)
        tensors.append(tensor)
    return tensors

# Load tensors from the 'clip' subfolder
clip_tensors = load_tensors('clip224')

# Dictionary to store the results
results = {}

# Compute the cosine similarity and average results
for subfolder in subfolders:
    # if subfolder == 'clip':
    #     continue
    
    other_tensors = load_tensors(subfolder)
    
    cosine_similarities = []
    
    for clip_tensor, other_tensor in tqdm(zip(clip_tensors, other_tensors), total=100, desc=f'Processing {subfolder}'):
        # Compute cosine similarity along the last dimension
        clip_tensor = normalize_feat(clip_tensor.cuda())
        other_tensor = normalize_feat(other_tensor.cuda())
        # similarity = F.cosine_similarity(clip_tensor, other_tensor, dim=-2)
        # # Mean the 576 dimension
        # # similarity_mean = similarity.mean(dim=0).item()
        # similarity_max = similarity.mean().item()
        # cosine_similarities.append(similarity_max)
        clip_tensor = clip_tensor.unsqueeze(0)  # shape: [1, 576, 1024]
        other_tensor = other_tensor.unsqueeze(1)  # shape: [576, 1, 1024]
        
        # Compute cosine similarity along the channel dimension
        similarity = F.cosine_similarity(other_tensor, clip_tensor, dim=-1)  # shape: [576, 576]
        
        # Find the max similarity for each vector in other_tensor
        max_similarity = similarity.max(dim=1).values  # shape: [576]
        cosine_similarities.append(max_similarity.mean().item())
    
    # Average across the 100 tensors
    avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
    results[subfolder] = avg_cosine_similarity

# Print the results
for subfolder, avg_similarity in results.items():
    print(f'Average cosine similarity between clip and {subfolder}: {avg_similarity}')
