import os
import time
import torch
import config
import numpy as np
import torch.nn.functional as F
import helper.constants as constants
from torch import nn
from attacks.ftrojan.ftrojan import impose
from helper.utils import get_FTROJAN_PARAM

def soft_threshold(x, thresh):
    return torch.sign(x) * F.relu(x.abs() - thresh)

def normalize_dictionary(D):
    norms = D.norm(dim=0, keepdim=True)
    return D / (norms + 1e-12)

def estimate_step_size(D_norm):
    if config.DICT_NORM == "Spectral norm":
        sigma = float(torch.linalg.norm(D_norm.detach(), ord=2))
    elif config.DICT_NORM == "Frobenius norm":
        sigma = float(torch.linalg.norm(D_norm.detach()))
    else:
        raise ValueError(f"Unsupported DICT_NORM={config.DICT_NORM}")

    L = sigma ** 2
    return float((1.0 / (L + 1e-8)) * config.DICT_ALPHA_ISTA)

def ista(X, D, lam, steps, step_size):
    n, _ = X.shape      # (n,p)
    _, k = D.shape      # (p,k)

    A = torch.zeros((n, k), device=X.device, dtype=X.dtype)
    for _ in range(steps):
        grad = (A @ D.t() - X) @ D
        A = soft_threshold(A - step_size * grad, lam * step_size)
        
    return A

def extract_nonoverlapping_patches_batch(images, patch_size_local):
    B, C, H, W = images.shape

    if H % patch_size_local != 0 or W % patch_size_local != 0:
        raise ValueError(
            f"Image size ({H}, {W}) must be divisible by patch_size={patch_size_local} "
            "for non-overlapping patch extraction."
        )

    unfold = nn.Unfold(kernel_size=patch_size_local, stride=patch_size_local).to(images.device)
    patches = unfold(images) 
    _, pdim, L = patches.shape

    patches = patches.permute(0, 2, 1).reshape(-1, pdim)  
    Hn = H // patch_size_local
    Wn = W // patch_size_local
    return patches, Hn, Wn

def extract_random_patches_from_batch(images, patch_size_local):
    B, C, H, W = images.shape
    device = images.device

    unfold = nn.Unfold(kernel_size=patch_size_local, stride=1).to(images.device)
    patches = unfold(images)
    _, pdim, _ = patches.shape

    patches = patches.permute(0, 2, 1).reshape(-1, pdim)
    total = patches.shape[0]
    if total == 0:
        raise ValueError("No patches available from this batch.")

    Hn = H // patch_size_local
    Wn = W // patch_size_local
    target_num = B * Hn * Wn

    if total >= target_num:
        idx = torch.randperm(total, device=device)[:target_num]
    else:
        idx = torch.randint(0, total, (target_num,), device=device)

    return patches[idx]

def reconstruct_batch_from_patches(patches_rec, B, Hn, Wn, patch_size_local):
    device_local = patches_rec.device
    L = Hn * Wn
    pdim = patches_rec.shape[1]

    patches_rec = patches_rec.reshape(B, L, pdim).permute(0, 2, 1).contiguous()
    fold = nn.Fold(
        output_size=(Hn * patch_size_local, Wn * patch_size_local),
        kernel_size=patch_size_local,
        stride=patch_size_local,
    ).to(device_local)
    return fold(patches_rec)

def get_batch_patches(train_iter, trainloader_for_patches):
    try:
        imgs = next(train_iter)[0].to(constants.DEVICE, non_blocking=True)
    except StopIteration:
        train_iter = iter(trainloader_for_patches)
        imgs = next(train_iter)[0].to(constants.DEVICE, non_blocking=True)

    if config.DICT_LEARN_OVERLAPPING_PATCHES:
        patches = extract_random_patches_from_batch(imgs, config.DICT_PATCH_SIZE)
    else:
        patches, _, _ = extract_nonoverlapping_patches_batch(imgs, config.DICT_PATCH_SIZE)

    return train_iter, patches

def dictionary_training_step(raw_D, X, ista_step):
    D_norm = normalize_dictionary(raw_D)

    if config.DICT_USE_MEAN_CENTERING:
        X_mean = X.mean(dim=1, keepdim=True)
        Xc = X - X_mean
    else:
        Xc = X

    A = ista(
        Xc,
        D_norm,
        config.DICT_L1_LAMBDA,
        config.DICT_ISTA_ITERS,
        ista_step,
    )
    Xr = A @ D_norm.t()
    loss_recon = ((Xc - Xr) ** 2).mean()

    return loss_recon

@torch.no_grad()
def apply_dictionary_to_images(images, D, ista_step):
    B, C, H, W = images.shape
    patches_all, Hn, Wn = extract_nonoverlapping_patches_batch(images, config.DICT_PATCH_SIZE)

    if config.DICT_USE_MEAN_CENTERING:
        patches_mean = patches_all.mean(dim=1, keepdim=True)
        patches_centered = patches_all - patches_mean
    else:
        patches_mean = torch.zeros_like(patches_all[:, :1])
        patches_centered = patches_all

    D = normalize_dictionary(D).to(images.device)

    codes_flat = ista(
        patches_centered.to(images.device),
        D,
        config.DICT_L1_LAMBDA,
        config.DICT_ISTA_ITERS,
        ista_step,
    )

    patches_rec = (codes_flat @ D.t()) + patches_mean.to(images.device)
    imgs_rec = reconstruct_batch_from_patches(patches_rec, B, Hn, Wn, config.DICT_PATCH_SIZE)

    L = Hn * Wn
    codes_map = (
        codes_flat.reshape(B, L, -1)
        .permute(0, 2, 1)
        .contiguous()
        .reshape(B, -1, Hn, Wn)
    )

    return codes_map, imgs_rec, codes_flat

def dictionary_frontend(images, D, ista_step):
    codes_map, imgs_rec, _ = apply_dictionary_to_images(images, D, ista_step)

    if config.DICT_OUTPUT == "reconstruction":
        return imgs_rec

    if config.DICT_OUTPUT == "codes":
        return codes_map

    raise ValueError(f"Unsupported DICT_OUTPUT={config.DICT_OUTPUT}")

def train_dictionary(trainloader_for_patches, validation_loader):
    sample_imgs = next(iter(trainloader_for_patches))[0].to(constants.DEVICE)
    _, in_channels, _, _ = sample_imgs.shape

    patch_dim = config.DICT_PATCH_SIZE * config.DICT_PATCH_SIZE * in_channels
    raw_D = nn.Parameter(torch.randn(patch_dim, config.DICT_NUM_ATOMS, device=constants.DEVICE) * 0.1)
    optimizer_D = torch.optim.Adam([raw_D], lr=config.DICT_LR)

    train_iter = iter(trainloader_for_patches)

    print("Training dictionary (unrolled ISTA)...")
    start_time = time.time()

    for epoch in range(1, config.DICT_EPOCHS + 1):
        D_norm = normalize_dictionary(raw_D)
        ista_step = estimate_step_size(D_norm)

        train_iter, X = get_batch_patches(train_iter, trainloader_for_patches)

        loss_recon = dictionary_training_step(raw_D, X, ista_step)

        optimizer_D.zero_grad()
        loss_recon.backward()
        optimizer_D.step()

    elapsed = time.time() - start_time
    print(f"Dictionary training done in {elapsed / 60:.2f} minutes")

    D_frozen = normalize_dictionary(raw_D).detach().clone().to(constants.DEVICE)
    ista_step_frozen = estimate_step_size(D_frozen)
    print("Using frozen ISTA step:", ista_step_frozen)
    return raw_D, D_frozen, ista_step_frozen