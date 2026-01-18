# -*- coding: utf-8 -*-
"""
@File   :  helpers.py
@Time   :  2026/01/18 14:37
@Author :  Yufan Liu
@Desc   :  Helper functions
"""
import torch
import random
import warnings
from einops import rearrange
from torch import Tensor


def masking(ratio: float, strategy):
    """Generate masks for reconstruction based on specified strategy.

    Args:
        ratio: Masking ratio
        strategy: Masking strategy: "random", "patch", "channel"
    """

    def random_mask(x, ratio=ratio):
        """Randomly mask 2D content across [C, N] plane.

        Args:
            x (Tensor): Input tensor of shape [B, C, N, D]
            ratio (float): Masking ratio between 0 and 1

        Returns:
            Tensor: Binary mask of shape [C, N] where 1 indicates masked positions
        """
        B, C, N, D = x.shape
        device = x.device

        noise = torch.rand(C, N, device=device)

        num_elements = C * N
        num_keep = int(num_elements * (1 - ratio))

        ids_shuffle = torch.argsort(noise.reshape(-1))

        mask = torch.ones([C, N], device=device)
        mask_flat = mask.reshape(-1)
        mask_flat[ids_shuffle[:num_keep]] = 0
        mask = mask_flat.reshape(C, N)

        return mask

    def patch_mask(x, ratio=ratio):
        """Mask all channels of selected patches.

        Args:
            x (Tensor): Input tensor of shape [B, C, N, D]
            ratio (float): Masking ratio between 0 and 1

        Returns:
            Tensor: Binary mask of shape [C, N] where 1 indicates masked patches
        """
        B, C, N, D = x.shape
        device = x.device

        noise = torch.rand(N, device=device)

        num_elements = N
        num_keep = int(num_elements * (1 - ratio))

        ids_shuffle = torch.argsort(noise)
        ids_keep = ids_shuffle[:num_keep]

        mask = torch.ones([N], device=device)
        mask[ids_keep] = 0

        mask = mask.unsqueeze(0).expand(C, -1)  # [C, N]

        return mask

    def specified_mask(x, channels):
        """Mask specified channels.

        Args:
            x (Tensor): Input tensor of shape [B, C, N, D]
            channels (list): List of channel indices to mask

        Returns:
            Tensor: Binary mask of shape [C, N] where 1 indicates masked specified channels
        """
        warnings.warn(f"{strategy} masking select, the ratio {ratio} will not be considered.")
        assert isinstance(channels, list), "channels must be a list"
        B, C, N, D = x.shape
        device = x.device
        mask = torch.zeros([C], device=device)
        mask[channels] = 1

        mask = mask.unsqueeze(1).expand(-1, N)  # [C, N]

        return mask

    strategies = {
        "random": random_mask,
        "patch": patch_mask,
        "specified": specified_mask,
    }

    return strategies[strategy]


def patchify(img: Tensor, patch_size=16, dense=False):
    """Convert input image into patches.

    Args:
        img (Tensor): Input image tensor of shape [B, H, W, C]
        patch_size (int): Size of each patch. Defaults to 16.
        dense (bool): Whether to use dense patching. Defaults to False.

    Returns:
        Tensor: Patched image of shape [B, H//P, W//P, P, P, C] where P is patch_size
    """
    B, H, W, C = img.shape  # H, W are set to 224 by default with processing
    P = patch_size
    assert H % P == 0, f"Image size must be divisible by patch size {P}"
    img = img.reshape(B, H // P, P, W // P, P, C)  # like [B, 14, 16, 14, 16, C]
    img = img.permute(0, 1, 3, 2, 4, 5)  # like [B, 14, 14, 16, 16, C]
    return img


def unpatchify(img: Tensor):
    """Reverse the patchify operation to reconstruct the original image.

    Args:
        img (Tensor): Patched image tensor of shape [B, H//P, W//P, P, P, C]

    Returns:
        Tensor: Reconstructed image of shape [B, H, W, C]
    """
    B, H, W, P, _, C = img.shape
    img = img.permute(0, 1, 3, 2, 4, 5)  # [B, H//P, P, W//P, P, C]
    img = img.reshape(B, H * P, W * P, C)  # [B, H, W, C]
    return img
