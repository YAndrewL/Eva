# -*- coding: utf-8 -*-
"""
@File   :  overlay.py
@Time   :  2025/12/10 09:13
@Author :  Yufan Liu
@Desc   :  Functions for overlaying prediction results and multiplex channels on wsi or patches

Functions:
    1. compose_overlay(data, channels, colors, ...)
       - Overlay multiple single-channel images onto one RGB image using specified colors
       - Supports flexible channel selection, color mapping, intensity adjustment, gamma correction,
         and automatic channel weighting to balance color dominance
       - Accepts data as numpy array or file path string
       - Returns RGB overlay image with shape HxWx3
"""

import numpy as np
from matplotlib.colors import to_rgb


def compose_overlay(
    data,
    channels,
    colors,
    intensity_multiplier=1.2,
    gamma_correction=1.0,
    normalize_per_channel=True,
    percentiles=(2.0, 98.0),
    weights=None,
):
    """
    Overlay multiple single-channel images onto one RGB image using specified colors.

    Args:
        data (np.ndarray): Input data with shape HxWxC
        channels (list): List of channel indices to overlay
        colors (list): Colors corresponding to channels (names or RGB tuples)
        intensity_multiplier (float): Multiplier for channel intensity
        gamma_correction (float): Gamma correction value
        normalize_per_channel (bool): Whether to normalize each channel independently
        percentiles (tuple): Tuple (low, high) for robust normalization per channel
        weights (list, str, None): Channel weights. If 'auto', weights are computed from
      each channel's intensity (p95) and the color's brightness (L1 of RGB),
      to reduce dominance of bright colors like yellow.

    Returns:
        np.ndarray: RGB overlay image with shape HxWx3
    """
    # Handle both file path and numpy array
    if isinstance(data, str):
        data = np.load(data)
    elif not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a numpy array or file path string, got {type(data)}")

    h, w, _ = data.shape
    rgb = np.zeros((h, w, 3), dtype=float)

    # Prepare per-channel intensity statistics for auto weighting
    if weights == "auto":
        p95_list = []
        for ch in channels:
            ch_data = data[:, :, ch].astype(float)
            p95 = float(np.percentile(ch_data, 95))
            p95_list.append(p95 if p95 > 1e-8 else 1e-8)
        median_p95 = float(np.median(p95_list)) if len(p95_list) > 0 else 1.0
        auto_weights = []
        for ch, color, p95 in zip(channels, colors, p95_list):
            rgb_color = to_rgb(color) if isinstance(color, str) else tuple(color)
            color_l1 = max(1e-8, float(rgb_color[0] + rgb_color[1] + rgb_color[2]))
            w_intensity = median_p95 / p95
            w_color = 1.0 / color_l1
            auto_weights.append(w_intensity * w_color)
        # Normalize weights to have mean 1
        mean_w = np.mean(auto_weights)
        weights = [float(w / mean_w) for w in auto_weights]

    if weights is None:
        weights = [1.0] * len(channels)

    for ch, color, wgt in zip(channels, colors, weights):
        ch_data = data[:, :, ch].astype(float)
        if normalize_per_channel:
            if percentiles is not None:
                lo, hi = np.percentile(ch_data, percentiles)
                if hi > lo:
                    ch_data = (ch_data - lo) / (hi - lo)
                else:
                    ch_data = ch_data - ch_data.min()
            else:
                vmin, vmax = ch_data.min(), ch_data.max()
                ch_data = (ch_data - vmin) / (vmax - vmin + 1e-8)
        ch_data = np.clip(ch_data * intensity_multiplier * wgt, 0, 1)
        ch_data = np.power(ch_data, gamma_correction)

        rgb_color = to_rgb(color) if isinstance(color, str) else tuple(color)
        for i in range(3):
            rgb[:, :, i] += ch_data * rgb_color[i]

    # Prevent over-saturation by normalizing where any channel exceeds 1
    max_val = np.maximum(1.0, rgb.max(axis=2, keepdims=True))
    rgb = rgb / max_val

    return np.clip(rgb, 0, 1)
