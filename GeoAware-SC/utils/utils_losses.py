import torch
import numpy as np
import torch.nn.functional as F
from utils.utils_geoware import permute_indices, flip_keypoints

def get_corr_map_loss(img1_desc, img2_desc, corr_map_net, img1_patch_idx, gt_flow, num_patches=60, img2_patch_idx = None):
    # img1_desc shape [1,3600,768]
    corr_map = torch.matmul(img1_desc, img2_desc.transpose(1,2)) # [1,3600,3600]
    corr_map = corr_map.reshape(1, num_patches, num_patches, num_patches, num_patches)
    # feed into network
    corr_map = corr_map_net(corr_map) # [1,60,60,2] 2 for x and y
    corr_map = corr_map.reshape(1, num_patches*num_patches, 2)
    # get the predicted flow
    predict_flow = corr_map[0, img1_patch_idx, :]
    EPE_loss = torch.norm(predict_flow - gt_flow, dim=-1).mean()

    return EPE_loss

def self_contrastive_loss(feat_map, instance_mask=None):
    """
    input: feat_map (B, C, H, W) mask (B, H', W')
    """
    B, C, H, W = feat_map.size()
    if instance_mask is not None:
        # interpolate the mask to the size of the feature map
        instance_mask = F.interpolate(instance_mask.cuda().unsqueeze(1).float(), size=(H, W), mode='bilinear')>0.5
        # mask out the feature map
        feat_map = feat_map * instance_mask
        # make where all zeros to be 1
        feat_map = feat_map + (~instance_mask)
    # Define neighborhood for local loss (8-neighborhood)
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
    local_loss = 0.0
    for i, j in offsets:
        # Shift feature map
        shifted_map = torch.roll(feat_map, shifts=(i, j), dims=(2, 3))
        # Compute the dot product
        dot_product = (feat_map * shifted_map).sum(dim=1)  # Sum along channel dimension
        # Only consider valid region (to avoid wrapping around difference)
        if i > 0:
            dot_product[:, :i, :] = 0
        if j > 0:
            dot_product[:, :, :j] = 0
        if i < 0:
            dot_product[:, i:, :] = 0
        if j < 0:
            dot_product[:, :, j:] = 0
        local_loss -= dot_product.mean()  # negative because we want to maximize the dot product for neighbors

    # For global loss, random sample non-neighbor pixels
    num_samples = H * W  # you can adjust this number based on your requirement
    idx_i = torch.randint(0, H, (num_samples,)).cuda()
    idx_j = torch.randint(0, W, (num_samples,)).cuda()
    idx_k = torch.randint(0, H, (num_samples,)).cuda()
    idx_l = torch.randint(0, W, (num_samples,)).cuda()

    # Ensure they are not neighbors
    mask = ((idx_k-idx_i).abs() > 1) | ((idx_l-idx_j).abs() > 1)
    if instance_mask is not None:
        mask = mask & instance_mask[0, 0, idx_i, idx_j] & instance_mask[0, 0, idx_k, idx_l]
    idx_i, idx_j, idx_k, idx_l = idx_i[mask], idx_j[mask], idx_k[mask], idx_l[mask]
    global_loss = 0.0
    for i, j, k, l in zip(idx_i, idx_j, idx_k, idx_l):
        dot_product = (feat_map[:, :, i, j] * feat_map[:, :, k, l]).sum(dim=1)
        global_loss += dot_product.mean()  # positive because we want to minimize the dot product for non-neighbors
    # Combine local and global losses
    lambda_factor = 0.1  # this can be adjusted based on cross-validation
    loss = local_loss + lambda_factor * global_loss
    return loss 

def get_logits(image_features, text_features, logit_scale):
    # Compute base logits
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logit_scale * text_features @ image_features.T

    return logits_per_image, logits_per_text

def cal_clip_loss(image_features, text_features, logit_scale, self_logit_scale = None):
    total_loss = 0

    device = image_features.device
    logits_per_image, logits_per_text = get_logits(image_features, text_features, logit_scale)
    labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
    total_loss += (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2

    return total_loss

def calculate_patch_indices_and_loss(args, kps_1, kps_2, desc_1, desc_2, scale_factor, num_patches, aggre_net, threshold, corr_map_net=None, device='cuda'):
    """
    Calculate patch indices and corresponding loss.
    
    Args:
    - kps_1, kps_2: Keypoints for the two images.
    - desc_1, desc_2: Descriptors for the two images.
    - scale_factor, num_patches: Parameters for calculating patch indices.
    - aggre_net: Aggregation network for calculating loss.
    - DENSE_OBJ: Boolean indicating if to use dense objective.
    - GAUSSIAN_AUGMENT: Gaussian augmentation factor.
    - threshold: Threshold for Gaussian augmentation.
    - corr_map_net: Correlation map network, required if CORR_MAP is True.
    - device: Device to use for tensor operations.
    
    Returns:
    - Loss calculated based on the provided parameters.
    """
    def get_patch_idx(scale_factor, num_patches, img1_y, img1_x):
        scaled_img1_y = scale_factor * img1_y
        scaled_img1_x = scale_factor * img1_x
        img1_y_patch = scaled_img1_y.astype(np.int32)
        img1_x_patch = scaled_img1_x.astype(np.int32)
        img1_patch_idx = num_patches * img1_y_patch + img1_x_patch
        if args.DENSE_OBJ:
            return scaled_img1_y, scaled_img1_x, img1_patch_idx
        else:
            return img1_y_patch, img1_x_patch, img1_patch_idx 
        
    # Calculate patch indices for both keypoints
    y1, x1 = kps_1[:, 1].numpy(), kps_1[:, 0].numpy()
    y2, x2 = kps_2[:, 1].numpy(), kps_2[:, 0].numpy()
    y_patch_1, x_patch_1, patch_idx_1 = get_patch_idx(scale_factor, num_patches, y1, x1)
    y_patch_2, x_patch_2, patch_idx_2 = get_patch_idx(scale_factor, num_patches, y2, x2)
    
    # Calculate loss based on whether correlation map is used
    if not args.DENSE_OBJ:
        desc_patch_1 = desc_1[0, patch_idx_1, :]
        desc_patch_2 = desc_2[0, patch_idx_2, :]
        loss = cal_clip_loss(desc_patch_1, desc_patch_2, aggre_net.logit_scale.exp(), self_logit_scale=aggre_net.self_logit_scale.exp())
    else:
        gt_flow = torch.stack([torch.tensor(x_patch_2) - torch.tensor(x_patch_1), torch.tensor(y_patch_2) - torch.tensor(y_patch_1)], dim=-1).to(device)
        if args.GAUSSIAN_AUGMENT > 0:
            std = args.GAUSSIAN_AUGMENT * threshold / 2
            noise = torch.randn_like(gt_flow, dtype=torch.float32) * std
            gt_flow += noise
        loss = get_corr_map_loss(desc_1, desc_2, corr_map_net, patch_idx_1, gt_flow, num_patches, img2_patch_idx=patch_idx_2)
    
    return loss

def calculate_loss(args, aggre_net, img1_kps, img2_kps, img1_desc, img2_desc, img1_threshold, img2_threshold, mask1, mask2, num_patches, device, raw_permute_list=None, img1_desc_flip=None, img2_desc_flip=None, corr_map_net=None):
    
    def get_patch_idx(args, scale_factor, num_patches, img1_y, img1_x):
        scaled_img1_y = scale_factor * img1_y
        scaled_img1_x = scale_factor * img1_x
        img1_y_patch = scaled_img1_y.astype(np.int32)
        img1_x_patch = scaled_img1_x.astype(np.int32)
        img1_patch_idx = num_patches * img1_y_patch + img1_x_patch
        if args.DENSE_OBJ:
            return scaled_img1_y, scaled_img1_x, img1_patch_idx
        else:
            return img1_y_patch, img1_x_patch, img1_patch_idx 
        
    vis = (img1_kps[:, 2] * img2_kps[:, 2]).bool()
    scale_factor = num_patches / args.ANNO_SIZE
    img1_y, img1_x = img1_kps[vis, 1].numpy(), img1_kps[vis, 0].numpy()
    img1_y_patch, img1_x_patch, img1_patch_idx = get_patch_idx(args, scale_factor, num_patches, img1_y, img1_x)
    img2_y, img2_x = img2_kps[vis, 1].numpy(), img2_kps[vis, 0].numpy()
    img2_y_patch, img2_x_patch, img2_patch_idx = get_patch_idx(args, scale_factor, num_patches, img2_y, img2_x)

    loss = cal_clip_loss(img1_desc[0, img1_patch_idx,:], img2_desc[0, img2_patch_idx,:], aggre_net.logit_scale.exp(), self_logit_scale=aggre_net.self_logit_scale.exp())

    if args.DENSE_OBJ > 0: # dense training objective loss
        flow_idx = img1_patch_idx
        flow_idx2 = img2_patch_idx
        gt_flow = torch.stack([torch.tensor(img2_x_patch) - torch.tensor(img1_x_patch), torch.tensor(img2_y_patch) - torch.tensor(img1_y_patch)], dim=-1).to(device)
        if args.GAUSSIAN_AUGMENT>0:
            std = args.GAUSSIAN_AUGMENT * img2_threshold / 2     # 2 sigma within the threshold
            noise = torch.randn_like(gt_flow, dtype=torch.float32) * std
            gt_flow = gt_flow + noise
        EPE_loss = get_corr_map_loss(img1_desc, img2_desc, corr_map_net, flow_idx, gt_flow, num_patches, img2_patch_idx=flow_idx2)
        loss += EPE_loss

    if args.ADAPT_FLIP > 0 or args.AUGMENT_SELF_FLIP > 0 or args.AUGMENT_DOUBLE_FLIP > 0:  # augment with flip
        loss = [loss]
        loss_weight = [1]
        
        permute_list = permute_indices(raw_permute_list)
        img1_kps_flip = flip_keypoints(img1_kps, args.ANNO_SIZE, permute_list)
        img2_kps_flip = flip_keypoints(img2_kps, args.ANNO_SIZE, permute_list)
        img1_kps = img1_kps[:len(permute_list), :]
        img2_kps = img2_kps[:len(permute_list), :]
        
        # Calculate losses for each augmentation type
        if args.ADAPT_FLIP > 0:
            vis_flip = img1_kps_flip[:, 2] * img2_kps[:, 2] > 0  # mutual visibility after flip
            if vis_flip.sum() > 0:
                loss_flip = calculate_patch_indices_and_loss(args, img1_kps_flip[vis_flip], img2_kps[vis_flip], img1_desc_flip, img2_desc, scale_factor, num_patches, aggre_net, img2_threshold, corr_map_net, device)
                loss.append(loss_flip)
                loss_weight.append(args.ADAPT_FLIP)

        if args.AUGMENT_DOUBLE_FLIP > 0:
            vis_double_flip = img1_kps_flip[:, 2] * img2_kps_flip[:, 2] > 0  # mutual visibility after flip
            if vis_double_flip.sum() > 0:
                loss_double_flip = calculate_patch_indices_and_loss(args, img1_kps_flip[vis_double_flip], img2_kps_flip[vis_double_flip], img1_desc_flip, img2_desc_flip, scale_factor, num_patches, aggre_net, img2_threshold, corr_map_net, device)
                loss.append(loss_double_flip)
                loss_weight.append(args.AUGMENT_DOUBLE_FLIP)
        
        if args.AUGMENT_SELF_FLIP > 0:
            vis_self_flip = img1_kps_flip[:, 2] * img1_kps[:, 2] > 0
            if vis_self_flip.sum() > 0:
                loss_self_flip = calculate_patch_indices_and_loss(args, img1_kps_flip[vis_self_flip], img1_kps[vis_self_flip], img1_desc_flip, img1_desc, scale_factor, num_patches, aggre_net, img1_threshold, corr_map_net, device)
                loss.append(loss_self_flip)
                loss_weight.append(args.AUGMENT_SELF_FLIP)
        
        # Aggregate losses
        loss = sum([l * w for l, w in zip(loss, loss_weight)]) / sum(loss_weight)

    if args.SELF_CONTRAST_WEIGHT>0:
        contrast_loss1 = self_contrastive_loss(img1_desc.permute(0,2,1).reshape(-1,args.PROJ_DIM,num_patches,num_patches), mask1.unsqueeze(0)) * args.SELF_CONTRAST_WEIGHT
        contrast_loss2 = self_contrastive_loss(img2_desc.permute(0,2,1).reshape(-1,args.PROJ_DIM,num_patches,num_patches), mask2.unsqueeze(0)) * args.SELF_CONTRAST_WEIGHT
        contrast_loss = (contrast_loss1 + contrast_loss2) / 2 * args.SELF_CONTRAST_WEIGHT
        loss += contrast_loss

    return loss