import torch
import numpy as np
import math 
from torch.nn import functional as F


def swap_elements_between_sets(indices, split_point, swap_count, device):
    """
    Swap swap_count elements randomly between two sets defined by split_point within each batch.

    indices: [B, L] index matrix
    split_point: boundary between sets A and B (num_mask or KEEP in the original code)
    """
    B, L = indices.shape

    max_swap = min(split_point, L - split_point)
    if swap_count > max_swap:
        #print(f"⚠️ Warning: swap_count ({swap_count}) exceeds the maximum ({max_swap}).")
        swap_count = max_swap

    if swap_count <= 0:
        return indices

    # Select random positions within set A (0 ~ split_point-1)
    # Use torch.rand().argsort() instead of torch.randperm(n) to enable batch processing
    rand_A_raw = torch.rand(B, split_point, device=device).argsort(dim=1)
    rand_A = rand_A_raw[:, :swap_count]

    # Select random positions within set B (split_point ~ L-1)
    rand_B_raw = torch.rand(B, L - split_point, device=device).argsort(dim=1)
    rand_B = rand_B_raw[:, :swap_count] + split_point

    # Extract values
    val_A = torch.gather(indices, 1, rand_A)
    val_B = torch.gather(indices, 1, rand_B)

    # Swap in one step using scatter_ (in-place swap may also be considered to minimize cloning)
    new_indices = indices.clone()
    new_indices.scatter_(1, rand_A, val_B)
    new_indices.scatter_(1, rand_B, val_A)
    
    return new_indices


def build_C_pca_batched(x_prime):
    """
    Normalized Cut approximation using PCA.

    Split each batch into 2 clusters based on the first principal component (PCA) and compute cluster centers.

    [PCA Partitioning]
    Uses principal component analysis (PCA) to identify the direction of maximum variance (first principal axis),
    then bi-partitions the data using the mean of each sample's projection onto that axis as a threshold.
    This method finds the single line that best explains (separates) the data in a high-dimensional feature space,
    and performs clustering using the centroid (mean) as the threshold.
    It is computationally cheaper than the Spectral method and well-suited for intuitive separation.

    Args:
        x_prime (torch.Tensor): [B, L, D] patch features
    Returns:
        C (torch.Tensor): [B, 2, D] normalized cluster centers
    """
    B, L, D = x_prime.shape
    dtype = x_prime.dtype

    # 1. L2 normalization of features
    x = F.normalize(x_prime, dim=-1)

    # 2. Compute the first principal component (PCA) (executed in float32)
    #    - Direction approximating the Fiedler vector of Normalized Cut
    with torch.cuda.amp.autocast(enabled=False):
        x_f32 = x.to(torch.float32)

        # q=1: compute only the first principal component per batch
        # U: [B, L, 1], S: [B, 1], V: [B, D, 1]
        _, _, V = torch.pca_lowrank(x_f32, q=1)

        # Projection direction of PCA (principal axis)
        direction = V  # [B, D, 1]

    # 3. Project onto the principal axis (cast back to original dtype)
    direction = direction.to(dtype)
    # [B, L, D] @ [B, D, 1] -> [B, L]
    proj = torch.bmm(x, direction).squeeze(-1)

    # 4. Bi-partition based on projection values
    #    - Use the mean of each batch as the threshold for bi-partition
    threshold = proj.mean(dim=1, keepdim=True)  # [B, 1]
    mask = (proj > threshold).to(dtype).unsqueeze(-1)  # [B, L, 1]

    # Weighted average per cluster (vectorized)
    c1 = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
    c2 = (x * (1.0 - mask)).sum(dim=1) / ((1.0 - mask).sum(dim=1) + 1e-6)

    # 5. Concatenate and normalize cluster centers
    C = torch.stack([c1, c2], dim=1) # [B, 2, D]
    return F.normalize(C, dim=-1)


def build_C_spectral_batched(tokens):
    """
    High-cost implementation following the paper.

    Compute the Fiedler vector via eigendecomposition of the graph Laplacian and derive 2 cluster centers.
    (High-cost approximation of the exact Normalized Cut solution.)

    [Spectral Partitioning / Fiedler Vector]
    A graph-theory-based method for bi-partitioning data by their similarity relations in the most balanced way.
    Excluding the smallest eigenvalue (0) of the graph Laplacian, the sign (positive/negative) of the
    eigenvector corresponding to the second smallest eigenvalue — the "Fiedler vector" — approximates
    the optimal binary partition of the data.
    In image analysis, it is used to separate objects from backgrounds based on inter-patch similarity.

    Args:
        tokens (torch.Tensor): [B, L, D] normalized features
    Returns:
        C (torch.Tensor): [B, 2, D] normalized cluster centers
    """
    # Similarity matrix M = X X^T
    M = torch.bmm(tokens, tokens.transpose(-2, -1))
    M = M + 1e-12  # Numerical stabilization

    # Laplacian L = D - M
    D_vec = M.sum(dim=2)
    L_mat = torch.diag_embed(D_vec) - M

    # Eigendecomposition
    _, eigvecs = torch.linalg.eigh(L_mat)

    # Second eigenvector (Fiedler vector)
    fiedler = eigvecs[:, :, 1] # Fiedler vector: [B, L]

    # 2. Normalized Cut approximation using the Fiedler vector
    mask_A = (fiedler >= 0).float().unsqueeze(-1)  # [B, L, 1]
    mask_B = 1.0 - mask_A                          # [B, L, 1]
    # 3. Compute cluster centers using masked mean (vectorized)
    c_A = (tokens * mask_A).sum(dim=1, keepdim=True) / (mask_A.sum(dim=1, keepdim=True) + 1e-8)
    c_B = (tokens * mask_B).sum(dim=1, keepdim=True) / (mask_B.sum(dim=1, keepdim=True) + 1e-8)
    centers = torch.cat([c_A, c_B], dim=1) # [B, 2, D]
    return torch.nn.functional.normalize(centers, p=2, dim=-1)

def swap_indices_with_hint(initial_mask_indices, initial_visible_indices, hint_ratio):
    device = initial_mask_indices.device
    B, num_mask = initial_mask_indices.shape
    num_unmask = int(initial_mask_indices.size(1) * hint_ratio)
    # A. Randomly select patches from M0 to retain as hints
    # Generate random permutation of [B, num_mask]
    rand_m = torch.argsort(torch.rand(B, num_mask, device=device), dim=1)
    # Split into those to be revealed as hints and those to remain masked
    m_to_v_idx = torch.gather(initial_mask_indices, 1, rand_m[:, :num_unmask])
    m_remain_idx = torch.gather(initial_mask_indices, 1, rand_m[:, num_unmask:])

    # B. Re-mask the same number of patches from the temporarily visible set [initial_visible + hints]
    v_temp_idx = torch.cat([initial_visible_indices, m_to_v_idx], dim=1) # [B, len_keep + num_unmask]

    # Randomly select for compensation
    rand_v = torch.argsort(torch.rand(B, v_temp_idx.size(1), device=device), dim=1)
    v_to_m_idx = torch.gather(v_temp_idx, 1, rand_v[:, :num_unmask])
    v_final_idx = torch.gather(v_temp_idx, 1, rand_v[:, num_unmask:])

    # 5. Compose final indices
    final_mask_indices = torch.cat([m_remain_idx, v_to_m_idx], dim=1)

    # Concatenate in order: [visible patches, masked patches]
    return torch.cat([v_final_idx, final_mask_indices], dim=1)

def object_centric_mask_with_hint(shape, mask_ratio, device, data, patch_size, hint_ratio, model=None, **kwargs):
    """
    Self-Guided Informed Masking (SGIM) implementation based on the SG-MAE paper.
    """
    if hint_ratio >= 1.0:
        return random_unstructured_mask(shape, mask_ratio, device, **kwargs)

    B, C, H, W = data.shape
    ph, pw = patch_size
    L = (H // ph) * (W // pw)
    len_keep = int(L * (1.0 - mask_ratio))
    num_mask = L - len_keep # num_mask in the original code

    # 1. Feature extraction using the encoder
    with torch.no_grad():
        # [B, L+1, D] -> [B, L, D]
        batch_x_all, *_ = model.forward_encoder(data.to(device), mask_ratio=0., return_layers=False)
        batch_x = batch_x_all[:, 1:, :]
        batch_x = torch.nn.functional.normalize(batch_x.float(), p=2, dim=-1)

        # 2. Normalized Cut approximation
        # centers = build_C_spectral_batched(batch_x)  # High-cost implementation following the paper
        centers = build_C_pca_batched(batch_x)         # Low-cost approximation

        # Similarity between each patch and the nearest cluster center
        # [B, L, D] @ [B, D, 2] -> [B, L, 2]
        scores = torch.bmm(batch_x, centers.transpose(1, 2))
        object_score = scores.max(dim=2).values # [B, L]

        # Sort scores in ascending order (lower scores go to the masked side)
        sorted_idx = object_score.argsort(dim=1)

    # Reorder as [visible, mask]; generate index matrix of [B, L]
    batch_indices = torch.cat([sorted_idx[:, num_mask:], sorted_idx[:, :num_mask]], dim=1)

    # 3. Index swap with hint
    if hint_ratio > 0:
        batch_indices = swap_indices_with_hint(sorted_idx[:, :num_mask], sorted_idx[:, num_mask:], hint_ratio)

    return batch_indices, len_keep


def random_weighted_mask_with_hint(shape, mask_ratio, device, data, patch_size, hint_ratio, metric='mad', probabilistic=True, **kwargs):
    """
    Dispersion-weighted masking (DWM) implementation.

    Stochastically generate a mask based on patch information content (variance, etc.) and return indices after hint processing.

    [Weighted Mask with Hint]
    Statistical measures of pixel values in each patch (variance or mean absolute deviation) are used as
    information weights, from which an initial mask is sampled with higher probability for more informative patches.
    Then, the hint process (Hint Ratio) reveals a portion of the initial mask and re-masks the same number
    from the initially visible patches, maintaining the total visible patch count while providing
    information from important regions as hints to the decoder.

    Args:
        shape (tuple): Unused in this function; defined for compatibility.
        mask_ratio (float): Final masking ratio (0.0 to 1.0).
        device (torch.device): Computation device.
        data (torch.Tensor): Input image data [B, C, H, W].
        patch_size (tuple): Patch size (ph, pw).
        hint_ratio (float): Fraction of initially masked patches to reveal as "hints".
        metric (str): Metric for weight computation ('variance' or 'mad').
    Returns:
        tuple:
            - batch_indices (torch.Tensor): Sorted index array of [B, L],
                                           composed in order: [visible patches, masked patches].
            - len_keep (int): Number of visible patches.
    """
    if hint_ratio >= 1.0:
        return random_unstructured_mask(shape, mask_ratio, device, **kwargs)

    B, C, H, W = data.shape
    ph, pw = patch_size
    device = data.device
    
    # 1. Patch splitting and weight (information content) computation
    num_ph, num_pw = H // ph, W // pw
    L = num_ph * num_pw
    len_keep = int(L * (1.0 - mask_ratio))
    num_mask = L - len_keep
    num_unmask = int(num_mask * hint_ratio)

    # Unfold image into patches [B, L, ph*pw]
    # (B, 1, H, W) -> (B, num_ph, ph, num_pw, pw) -> (B, num_ph, num_pw, ph, pw) -> (B, L, ph*pw)
    patches = data.view(B, 1, num_ph, ph, num_pw, pw).transpose(3, 4).reshape(B, L, -1)

    # Compute local complexity per patch based on the specified metric
    if metric == 'variance':
        weights = torch.var(patches, dim=2)
    elif metric == 'mad':
        weights = torch.mean(torch.abs(patches - torch.mean(patches, dim=2, keepdim=True)), dim=2)
    else:
        raise ValueError("Invalid metric")

    # 2. Determine initial mask via weighted sampling
    # Avoid zero probability
    probabilities = weights + 1e-12  # Add small epsilon to avoid zero-probability errors
    if probabilistic:
        # Sample mask targets without replacement with probability proportional to weights [B, num_mask]
        initial_mask_indices = torch.multinomial(probabilities, num_mask, replacement=False)
    else:
        # Deterministically select the top num_mask indices with the highest weights
        _, initial_mask_indices = torch.topk(probabilities, k=num_mask, dim=1, largest=True, sorted=False)

    # 3. Extract the initial visible set from all indices
    all_indices = torch.arange(L, device=device).expand(B, L)
    mask_bool = torch.zeros(B, L, device=device, dtype=torch.bool)
    mask_bool.scatter_(1, initial_mask_indices, True)  # Set initial mask positions to True

    # Extract unmasked positions (False) to obtain initial visible indices [B, len_keep]
    initial_visible_indices = all_indices[~mask_bool].view(B, len_keep)

    # 4. Hint processing (mask→visible) and compensation (visible→mask)

    # A. Randomly select patches from the initial mask (M0) to reveal as hints
    rand_m = torch.argsort(torch.rand(B, num_mask, device=device), dim=1)
    m_to_v_idx = torch.gather(initial_mask_indices, 1, rand_m[:, :num_unmask])  # Revealed as hints
    m_remain_idx = torch.gather(initial_mask_indices, 1, rand_m[:, num_unmask:])  # Remain masked

    # B. Re-mask the same number from the expanded visible set [initial_visible + hints] to maintain total count
    v_temp_idx = torch.cat([initial_visible_indices, m_to_v_idx], dim=1) # [B, len_keep + num_unmask]

    # Randomly select candidates for re-masking
    rand_v = torch.argsort(torch.rand(B, v_temp_idx.size(1), device=device), dim=1)
    v_to_m_idx = torch.gather(v_temp_idx, 1, rand_v[:, :num_unmask])  # Compensation masks
    v_final_idx = torch.gather(v_temp_idx, 1, rand_v[:, num_unmask:])  # Final visible set

    # 5. Construct the final index array
    final_mask_indices = torch.cat([m_remain_idx, v_to_m_idx], dim=1)

    # Concatenate as [visible, mask] to conform to the input format of MAE and similar models
    batch_indices = torch.cat([v_final_idx, final_mask_indices], dim=1)

    return batch_indices, len_keep

def random_weighted_mask_var_with_hint(shape, mask_ratio, device, data, patch_size, hint_ratio, **kwargs):
    return random_weighted_mask_with_hint(shape, mask_ratio, device, data, patch_size, hint_ratio, metric='variance', **kwargs)

def determ_weighted_mask_with_hint(shape, mask_ratio, device, data, patch_size, hint_ratio, **kwargs):
    # DWM variant for ablation -- what if we purely follow the patch-wise dispersion?
    return random_weighted_mask_with_hint(shape, mask_ratio, device, data, patch_size, hint_ratio, probabilistic=False, **kwargs)


def random_unstructured_mask(shape, mask_ratio, device, **kwargs):
    B, F, T = shape # Batch, Freq bins, and Time frames; equivalent to Batch, Height, and Width for the image.
    L = F * T
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(B, L, device=device)  # noise in [0, 1]
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    return ids_shuffle, len_keep


def random_structured_mask(shape, mask_ratio, device, **kwargs):
    """Random structured masking for training in audio tasks."""
    B, F, T = shape

    # We want true random freq/time masking but need to make the number of masks consistent among samples.
    # We impose a constraint that the number of freq/time masks be consistent across samples while leaving it open where we mask.
    NF = int(F * (mask_ratio + 1./F) * np.random.rand())
    NF = min(F - 1, NF) # prevent masking all freq. bins.
    mask_ratio = max(mask_ratio + (.5/T) - (NF/F), 0.)
    NT = int(T*mask_ratio)

    # Make mask for each batch sample.
    mask = torch.zeros((B, F, T), dtype=torch.int, device=device)
    for b in range(B):
        mask[b, torch.randperm(F)[:NF]] = 1
    for b in range(B):
        mask[b, :, torch.randperm(T)[:NT]] = 1

    ids_shuffle = torch.argsort(mask.view(B, -1), descending=True)
    len_keep = (mask[0] == 0).sum()
    # print(len_keep, mask[:2])
    return ids_shuffle, len_keep


def make_one_1dmask(total_frames=800//2, mask_ratio=0.6, one_mask_frames=20):
    M = int(total_frames * mask_ratio)
    mask = np.zeros(total_frames, dtype=np.int8)
    # Mask frames more than M, number of frames to mask.
    while mask.sum() < M:
        i = np.random.randint(low=0, high=total_frames)
        mask[i:i+one_mask_frames] = 1
    # Unmask frames from the tail to adjust total masked frames to M.
    n_unmask = mask.sum() - M
    if n_unmask > 0:
        j_unmask = np.where(mask == 1)[0][-n_unmask:]
        mask[j_unmask] = 0
    return mask

def random_1d_mask(shape, mask_ratio=0.6, device='cuda', one_mask_frames=20//2, **kwargs):
    B, F, T = shape
    mask = np.zeros((B, T), dtype=np.int8)
    for i in range(B):
        mask[i] = make_one_1dmask(total_frames=T, mask_ratio=mask_ratio, one_mask_frames=one_mask_frames)
    mask = np.tile(mask, (1, F))
    mask = torch.tensor(mask).to(device)

    ids_shuffle = torch.argsort(mask.view(B, -1), dim=1)
    len_keep = (mask[0] == 0).sum()
    return ids_shuffle, len_keep


# Borrowed from https://github.com/cwx-worst-one/EAT/blob/06c1c34e2afd8dc35297e6d3815d31dbeec9d372/utils/data_utils.py#L211
def compute_block_mask_2d(      
    shape,
    mask_prob: float,
    mask_length: int,
    mask_prob_adjust: float = 0,
    inverse_mask: bool = False,
    require_same_masks: bool = True,
    expand_adjcent: bool = False,
    mask_dropout: float = 0,
    non_overlapping: bool = False,
    img_shape = None,
    flexible_mask: bool = False,
) -> torch.Tensor:

    assert mask_length > 1

    B, L = shape

    d = (int(L**0.5),int(L**0.5))
    
    if img_shape:
        d = (img_shape[0],img_shape[1])
        
    if flexible_mask:
        index = np.random.randint(0,3)
        block_size_options = np.array([(6, 4), (5, 5), (8, 3)])
        block_size = block_size_options[index]

    if inverse_mask:
        mask_prob = 1 - mask_prob
        
    if flexible_mask:
        mask = torch.zeros((B, d[0], d[1]))
        mask_inds = torch.randint(
            0,
            L,  
            size=(
                B,
                int(
                    L
                    * ((mask_prob + mask_prob_adjust) / (block_size[0]*block_size[1]))
                    * (1 + mask_dropout)
                ),
            ),
        )
        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)

        inds = ([], [], [])

        offset = mask_length // 2
        for i in range(block_size[0]):
            for j in range(block_size[1]):
                k1 = i - offset
                k2 = j - offset
                inds[0].append(centers[0])
                inds[1].append(centers[1] + k1)
                inds[2].append(centers[2] + k2)

        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=d[0] - 1)
        i2 = torch.cat(inds[2]).clamp_(min=0, max=d[1] - 1)

        mask[(i0, i1, i2)] = 1

    elif non_overlapping:
        sz = math.ceil(d[0] / mask_length)
        inp_len = sz * sz

        inp = torch.zeros((B, 1, sz, sz))
        w = torch.ones((1, 1, mask_length, mask_length))

        mask_inds = torch.multinomial(
            1 - inp.view(B, -1),
            int(inp_len * (mask_prob + mask_prob_adjust) * (1 + mask_dropout)),
            replacement=False,
        )
        inp.view(B, -1).scatter_(1, mask_inds, 1)

        mask = torch.nn.functional.conv_transpose2d(inp, w, stride=mask_length).squeeze(
            1
        )
        if mask.size(-1) > d[0]:
            mask = mask[..., :d, :d]
    else:
        mask = torch.zeros((B, d[0], d[1]))
        mask_inds = torch.randint(
            0,
            L,  
            size=(
                B,
                int(
                    L
                    * ((mask_prob + mask_prob_adjust) / mask_length**2)
                    * (1 + mask_dropout)
                ),
            ),
        )
        mask.view(B, -1).scatter_(1, mask_inds, 1)
        centers = mask.nonzero(as_tuple=True)

        inds = ([], [], [])

        offset = mask_length // 2
        for i in range(mask_length):
            for j in range(mask_length):
                k1 = i - offset
                k2 = j - offset
                inds[0].append(centers[0])
                inds[1].append(centers[1] + k1)
                inds[2].append(centers[2] + k2)

        i0 = torch.cat(inds[0])
        i1 = torch.cat(inds[1]).clamp_(min=0, max=d[0] - 1)
        i2 = torch.cat(inds[2]).clamp_(min=0, max=d[1] - 1)

        mask[(i0, i1, i2)] = 1

    def get_nbs(b, m, w):
        all_nbs = torch.nn.functional.conv2d(m.unsqueeze(1), w, padding="same")
        all_nbs = all_nbs.clamp_max_(1).view(b, -1)
        return all_nbs

    if require_same_masks and expand_adjcent:
        w = torch.zeros((1, 1, 3, 3))
        w[..., 0, 1] = 1
        w[..., 2, 1] = 1
        w[..., 1, 0] = 1
        w[..., 1, 2] = 1

        all_nbs = get_nbs(B, mask, w)

    mask = mask.reshape(B, -1)

    if require_same_masks:
        n_masks = mask.sum(dim=-1)
        final_target_len = int(L * (mask_prob))
        target_len = int(final_target_len * (1 + mask_dropout))

        for i in range(len(mask)):
            n = n_masks[i]
            m = mask[i]
            r = 0
            while expand_adjcent and n < target_len:
                if r == 0:
                    nbs = all_nbs[i]
                else:
                    nbs = get_nbs(1, m.view(1, d[0], d[1]), w).flatten()

                cands = (1 - m + nbs) > 1
                cand_sz = int(cands.sum().item())

                assert cand_sz > 0, f"{nbs} {cand_sz}"

                to_mask = torch.multinomial(
                    cands.float(), min(cand_sz, int(target_len - n)), replacement=False
                )
                m[to_mask] = 1
                assert to_mask.numel() > 0
                n += to_mask.numel()
                r += 1

            if n > final_target_len:
                to_unmask = torch.multinomial(
                    m, int(n - final_target_len), replacement=False
                )
                m[to_unmask] = 0
            elif n < final_target_len:
                to_mask = torch.multinomial(
                    (1 - m), int(final_target_len - n), replacement=False
                )
                m[to_mask] = 1

    if inverse_mask:
        mask = 1 - mask

    return mask


def data2vec2_block_masking_wrapper(shape, mask_ratio, device, hint_ratio, **kwargs):
    if hint_ratio >= 1.0:  # for compatibility with other masking behavior
        return random_unstructured_mask(shape, mask_ratio, device, **kwargs)

    B, Nf, Nt = shape
    binary_mask = compute_block_mask_2d(           
        shape=[B, Nf * Nt],  # (B, L)
        mask_prob=mask_ratio,
        mask_length=5,
        mask_prob_adjust=0.07,
        inverse_mask=True,
        require_same_masks=True,
        expand_adjcent=False,
        mask_dropout=0,
        non_overlapping=False,
        img_shape=[Nf, Nt],
        flexible_mask=False
    )

    B, L = binary_mask.shape
    
    num_keep = int((binary_mask == 0).sum(dim=1).min().item())
    noise = torch.rand(B, L, device=device) 
    sort_priority = binary_mask.to(device) + noise
    ids_shuffle = torch.argsort(sort_priority, dim=1)
    
    return ids_shuffle, num_keep
