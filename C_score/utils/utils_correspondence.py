import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

def convert_to_binary_mask(img_path, threshold=127, angle=None):
    # Open the image using Pillow
    img = Image.open(img_path).convert('L')  # Convert to grayscale

    # Rotate the image if angle is specified
    if angle is not None:
        img = img.rotate(angle)

    # Convert the image to a PyTorch tensor
    img_tensor = torch.from_numpy(np.array(img))

    # Create a binary mask
    binary_mask = (img_tensor > threshold).float() # Convert to binary mask, type float 32

    return binary_mask

def get_distance(feature1, feature2, mask1, mask2, RESOLUTION=64):
    src_feature_reshaped = feature1.squeeze().permute(1,0).reshape(1,-1,60,60).cuda()
    tgt_feature_reshaped = feature2.squeeze().permute(1,0).reshape(1,-1,60,60).cuda()

    patch_size = RESOLUTION
    resized_src_mask = F.interpolate(mask1.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size), mode='nearest').squeeze().cuda()
    resized_tgt_mask = F.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size), mode='nearest').squeeze().cuda()
    src_feature_upsampled = F.interpolate(src_feature_reshaped, size=(patch_size, patch_size), mode='bilinear').squeeze()
    tgt_feature_upsampled = F.interpolate(tgt_feature_reshaped, size=(patch_size, patch_size), mode='bilinear').squeeze()
    # mask the feature
    src_feature_upsampled = src_feature_upsampled * resized_src_mask.repeat(src_feature_upsampled.shape[0],1,1)
    tgt_feature_upsampled = tgt_feature_upsampled * resized_tgt_mask.repeat(src_feature_upsampled.shape[0],1,1)
    # Set the masked area to a very small number
    src_feature_upsampled[src_feature_upsampled == 0] = -100000
    tgt_feature_upsampled[tgt_feature_upsampled == 0] = -100000
    # Calculate the cosine similarity between src_feature and tgt_feature
    src_features_2d=src_feature_upsampled.reshape(src_feature_upsampled.shape[0],-1).permute(1,0)
    tgt_features_2d=tgt_feature_upsampled.reshape(tgt_feature_upsampled.shape[0],-1).permute(1,0)

    min_dist = []
    for patch_idx in range(patch_size*patch_size):
        # If the patch is in the resized_src_mask_out_layers, find the corresponding patch in the target_output and swap them
        if resized_src_mask[patch_idx // patch_size, patch_idx % patch_size] == 1:
            # Find the corresponding patch with the highest cosine similarity
            distances = torch.linalg.norm(tgt_features_2d - src_features_2d[patch_idx], dim=1)
            tgt_patch_idx = torch.argmin(distances)
            min_dist.append(distances[tgt_patch_idx].item())
    min_dist = torch.tensor(min_dist)
    min_dist = min_dist.mean()

    return min_dist

def get_distance_mutual_nn(feature1, feature2):
    feature1 = feature1.cuda()  # (1,3600,3968)
    feature2 = feature2.cuda()  # (1,3600,3968)

    distances_1to2 = torch.cdist(feature1, feature2)[0] # (3600,3600)

    nearest_patch_indices_1to2 = torch.argmin(distances_1to2, dim=1)
    nearest_patch_indices_2to1 = torch.argmin(distances_1to2, dim=0)

    # get the mutual nearest neighbors
    mutual_nn_1to2 = torch.zeros_like(nearest_patch_indices_1to2)
    mutual_nn_2to1 = torch.zeros_like(nearest_patch_indices_2to1)
    for i in range(len(nearest_patch_indices_1to2)):
        if nearest_patch_indices_2to1[nearest_patch_indices_1to2[i]] == i:
            mutual_nn_1to2[i] = 1
            mutual_nn_2to1[nearest_patch_indices_1to2[i]] = 1

    # get the average distance of the mutual nearest neighbors
    avg_distance_1to2 = torch.min(distances_1to2, dim=1)[0][mutual_nn_1to2==1].mean()
    return avg_distance_1to2

def resize(img, target_res=224, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)

def pairwise_sim(x: torch.Tensor, y: torch.Tensor, p=2, normalize=False) -> torch.Tensor:
    # compute similarity based on euclidean distances
    if normalize:
        x = torch.nn.functional.normalize(x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)
    result_list=[]
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)
        result_list.append(torch.nn.PairwiseDistance(p=p)(token, y)*(-1))
    return torch.stack(result_list, dim=2)


def co_pca(features1, features2, dim=[128,128,128]):
    
    processed_features1 = {}
    processed_features2 = {}
    s5_size = features1['s5'].shape[-1]
    s4_size = features1['s4'].shape[-1]
    s3_size = features1['s3'].shape[-1]
    # Get the feature tensors
    s5_1 = features1['s5'].reshape(features1['s5'].shape[0], features1['s5'].shape[1], -1)
    s4_1 = features1['s4'].reshape(features1['s4'].shape[0], features1['s4'].shape[1], -1)
    s3_1 = features1['s3'].reshape(features1['s3'].shape[0], features1['s3'].shape[1], -1)

    s5_2 = features2['s5'].reshape(features2['s5'].shape[0], features2['s5'].shape[1], -1)
    s4_2 = features2['s4'].reshape(features2['s4'].shape[0], features2['s4'].shape[1], -1)
    s3_2 = features2['s3'].reshape(features2['s3'].shape[0], features2['s3'].shape[1], -1)
    # Define the target dimensions
    target_dims = {'s5': dim[0], 's4': dim[1], 's3': dim[2]}

    # Compute the PCA
    for name, tensors in zip(['s5', 's4', 's3'], [[s5_1, s5_2], [s4_1, s4_2], [s3_1, s3_2]]):
        target_dim = target_dims[name]

        # Concatenate the features
        features = torch.cat(tensors, dim=-1) # along the spatial dimension
        features = features.permute(0, 2, 1) # Bx(t_x+t_y)x(d)

        # Compute the PCA
        # pca = faiss.PCAMatrix(features.shape[-1], target_dim)

        # Train the PCA
        # pca.train(features[0].cpu().numpy())

        # Apply the PCA
        # features = pca.apply(features[0].cpu().numpy()) # (t_x+t_y)x(d)

        # convert to tensor
        # features = torch.tensor(features, device=features1['s5'].device).unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        
        
        # equivalent to the above, pytorch implementation
        mean = torch.mean(features[0], dim=0, keepdim=True)
        centered_features = features[0] - mean
        U, S, V = torch.pca_lowrank(centered_features, q=target_dim)
        reduced_features = torch.matmul(centered_features, V[:, :target_dim]) # (t_x+t_y)x(d)
        features = reduced_features.unsqueeze(0).permute(0, 2, 1) # Bx(d)x(t_x+t_y)
        

        # Split the features
        processed_features1[name] = features[:, :, :features.shape[-1] // 2] # Bx(d)x(t_x)
        processed_features2[name] = features[:, :, features.shape[-1] // 2:] # Bx(d)x(t_y)

    # reshape the features
    processed_features1['s5']=processed_features1['s5'].reshape(processed_features1['s5'].shape[0], -1, s5_size, s5_size)
    processed_features1['s4']=processed_features1['s4'].reshape(processed_features1['s4'].shape[0], -1, s4_size, s4_size)
    processed_features1['s3']=processed_features1['s3'].reshape(processed_features1['s3'].shape[0], -1, s3_size, s3_size)

    processed_features2['s5']=processed_features2['s5'].reshape(processed_features2['s5'].shape[0], -1, s5_size, s5_size)
    processed_features2['s4']=processed_features2['s4'].reshape(processed_features2['s4'].shape[0], -1, s4_size, s4_size)
    processed_features2['s3']=processed_features2['s3'].reshape(processed_features2['s3'].shape[0], -1, s3_size, s3_size)

    # Upsample s5 spatially by a factor of 2
    processed_features1['s5'] = F.interpolate(processed_features1['s5'], size=(processed_features1['s4'].shape[-2:]), mode='bilinear', align_corners=False)
    processed_features2['s5'] = F.interpolate(processed_features2['s5'], size=(processed_features2['s4'].shape[-2:]), mode='bilinear', align_corners=False)

    # Concatenate upsampled_s5 and s4 to create a new s5
    processed_features1['s5'] = torch.cat([processed_features1['s4'], processed_features1['s5']], dim=1)
    processed_features2['s5'] = torch.cat([processed_features2['s4'], processed_features2['s5']], dim=1)

    # Set s3 as the new s4
    processed_features1['s4'] = processed_features1['s3']
    processed_features2['s4'] = processed_features2['s3']

    # Remove s3 from the features dictionary
    processed_features1.pop('s3')
    processed_features2.pop('s3')

    # current order are layer 8, 5, 2
    features1_gether_s4_s5 = torch.cat([processed_features1['s4'], F.interpolate(processed_features1['s5'], size=(processed_features1['s4'].shape[-2:]), mode='bilinear')], dim=1)
    features2_gether_s4_s5 = torch.cat([processed_features2['s4'], F.interpolate(processed_features2['s5'], size=(processed_features2['s4'].shape[-2:]), mode='bilinear')], dim=1)

    return features1_gether_s4_s5, features2_gether_s4_s5

def softmax_with_temperature(x, beta, d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

def soft_argmax(corr, beta=0.02):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    # input shape : (B, H_t * W_t, H_s , W_s) e.g., (B, 32*32, 32, 32)
    b, htwt, h, w = corr.size()
    ht, wt = int(np.sqrt(htwt)), int(np.sqrt(htwt))
    x_normal = np.linspace(-1,1,w)
    x_normal = torch.tensor(x_normal, device=corr.device).float()
    y_normal = np.linspace(-1,1,h)
    y_normal = torch.tensor(y_normal, device=corr.device).float()
    
    corr = softmax_with_temperature(corr, beta=beta, d=1) # (B, H_t * W_t, H_s , W_s)
    corr = corr.view(-1,ht,wt,h,w) # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
    x_normal = x_normal.expand(b,w)
    x_normal = x_normal.view(b,w,1,1)
    grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    
    grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
    y_normal = y_normal.expand(b,h)
    y_normal = y_normal.view(b,h,1,1)
    grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    return grid_x, grid_y

def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise

    # xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    # yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    # xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    # yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    # grid = torch.cat((xx,yy),1).float()

    # if mapping.is_cuda:
    #     grid = grid.cuda()
    # mapping = mapping - grid
    flow = mapping
    return flow

def apply_gaussian_kernel(corr, sigma=5):
    b, hw, h, w = corr.size() # b, h_t*w_t, h_s, w_s

    idx = corr.max(dim=1)[1] # b x h x w    get maximum value along channel
    idx_y = (idx // w).view(b, 1, 1, h, w).float()
    idx_x = (idx % w).view(b, 1, 1, h, w).float()
    x = np.linspace(0,59,60)
    x = torch.tensor(x, dtype=torch.float, requires_grad=False).to(corr.device)
    y = np.linspace(0,59,60)
    y = torch.tensor(y, dtype=torch.float, requires_grad=False).to(corr.device)
    x = x.view(1,1,w,1,1).expand(b, 1, w, h, w)
    y = y.view(1,h,1,1,1).expand(b, h, 1, h, w)

    gauss_kernel = torch.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * sigma**2))
    gauss_kernel = gauss_kernel.view(b, hw, h, w)

    return gauss_kernel * corr

def get_flow(corr, flow_window=0, num_patches=60):
    # corr: (H_s * W_s, H_t * W_t)
    hsws, htwt = corr.size()
    hs, ws = int(np.sqrt(hsws)), int(np.sqrt(hsws))
    ht, wt = int(np.sqrt(htwt)), int(np.sqrt(htwt))

    if flow_window > 0:  # zero out the corr_map outside the window
        # get the argmax
        max_index_flatten = torch.argmax(corr, dim=-1)
        max_index_x = max_index_flatten % num_patches  # (H_s * W_s, )
        max_index_y = max_index_flatten // num_patches  # (H_s * W_s, )
        corr = corr.view(-1, num_patches, num_patches)

        # Prepare offsets
        offset_range = torch.arange(-flow_window, flow_window + 1, device=corr.device)
        offset_x, offset_y = torch.meshgrid(offset_range, offset_range, indexing='ij')
        offset_x, offset_y = offset_x.flatten(), offset_y.flatten()

        # Compute window mask without loops
        window_positions_x = (max_index_x[:, None] + offset_x[None, :]).clamp(0, num_patches - 1)
        window_positions_y = (max_index_y[:, None] + offset_y[None, :]).clamp(0, num_patches - 1)

        # Create indices for gathering values
        batch_indices = torch.arange(corr.shape[0], device=corr.device)[:, None]
        
        # Using advanced indexing to create the window mask
        window_mask = torch.zeros_like(corr)
        window_mask[batch_indices, window_positions_y, window_positions_x] = 1

        # Apply window mask
        corr = corr * window_mask
    elif flow_window<0: #kernel soft_argmax
        corr = corr.permute(1,0).view(1,num_patches**2,num_patches,num_patches)
        corr = apply_gaussian_kernel(corr, sigma=-flow_window)
        corr = corr.view(num_patches**2,num_patches**2).permute(1,0)
    x = corr.view(-1,ht,wt,hsws)
    grid_x, grid_y = soft_argmax(x.permute(0, 3, 1, 2))
    x = torch.cat((grid_x, grid_y), dim=1)
    x = unnormalise_and_convert_mapping_to_flow(x) # (B, 2, H, W)
    # x = self.output_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    return x.permute(0, 2, 3, 1) # (B, H, W, 2)

def load_img_and_kps(idx, files, kps, img_size=224, edge=False):
    img = Image.open(files[idx]).convert('RGB')
    img = resize(img, img_size, resize=True, to_pil=True, edge=edge)
    img_kps = kps[idx]
    return img, img_kps

def calculate_keypoint_transformation(args, img1_desc, img2_desc, img1_patch_idx, num_patches):
    """
    Calculate the keypoint transformation from image 1 to image 2.

    Args:
        img1_desc (torch.Tensor): The patch descriptors for image 1.
        img2_desc (torch.Tensor): The patch descriptors for image 2.
        img1_patch_idx (torch.Tensor): The patch indices for the keypoints in image 1.
        num_patches (int): The number of patches the image is divided into.
        args (Namespace): The arguments containing method-specific parameters.

    Returns:
        torch.Tensor: The transformed keypoints from image 1 to image 2.
    """
    # Calculate similarity matrix
    sim_1_to_2 = torch.matmul(img1_desc, img2_desc.permute(0, 2, 1))[0]  # [3600, 3600]
    sim_1_to_2_idxed = sim_1_to_2[img1_patch_idx]
    anno_stride = args.ANNO_SIZE / num_patches

    if args.SOFT_EVAL:
        # Calculate flow if soft evaluation is enabled
        flow = get_flow(sim_1_to_2, args.SOFT_EVAL_WINDOW, num_patches)
        flow_flatten = flow.reshape(-1, 2)
        flow_idxed = flow_flatten[img1_patch_idx]
        nn_y_patch, nn_x_patch = flow_idxed[:, 1].clamp(0, num_patches - 1), flow_idxed[:, 0].clamp(0, num_patches - 1)
    else:
        # Find nearest neighbors if soft evaluation is not enabled
        _, nn_1_to_2 = torch.max(sim_1_to_2_idxed, dim=-1)
        nn_y_patch, nn_x_patch = nn_1_to_2 // num_patches, nn_1_to_2 % num_patches

    # Calculate the transformed keypoints' positions
    nn_x = nn_x_patch * anno_stride + anno_stride // 2
    nn_y = nn_y_patch * anno_stride + anno_stride // 2

    # Stack the transformed keypoints
    kps_1_to_2 = torch.stack([nn_x, nn_y]).permute(1, 0)

    return kps_1_to_2

def kpts_to_patch_idx(args, img1_kps, num_patches):
    img1_y, img1_x = img1_kps[:, 1].numpy(), img1_kps[:, 0].numpy()
    img1_y_patch = (num_patches / args.ANNO_SIZE * img1_y).astype(np.int32)
    img1_x_patch = (num_patches / args.ANNO_SIZE * img1_x).astype(np.int32)
    img1_patch_idx = num_patches * img1_y_patch + img1_x_patch
    return img1_patch_idx