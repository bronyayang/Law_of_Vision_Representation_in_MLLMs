import os
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path):
    # Load the tensor
    tensor = torch.load(file_path)
    
    # Squeeze the tensor to remove the singleton dimension if necessary
    if tensor.dim() > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    # Save the tensor back to its original location
    torch.save(tensor, file_path)

def adjust_tensor_shapes(root_dir):
    pt_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pt'):
                pt_files.append(os.path.join(subdir, file))

    # Using ThreadPoolExecutor to parallelize the operation
    with ThreadPoolExecutor() as executor:
        # Wrapping the executor.map call with tqdm for a progress bar
        list(tqdm(executor.map(process_file, pt_files), total=len(pt_files), desc="Adjusting tensors"))

# Replace '/path/to/your/folder' with the path to your folder
adjust_tensor_shapes('/mnt/bn/shijiaynas/LLaVA_finetune_sd1.5')
