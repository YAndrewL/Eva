#!/usr/bin/env python3

import importlib.util
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


GIGATIME_MARKERS = [
    "DAPI",
    "TRITC",
    "Cy5",
    "PD-1_1:200",
    "CD14",
    "CD4",
    "T-bet",
    "CD34",
    "CD68_1:100",
    "CD16",
    "CD11c",
    "CD138",
    "CD20",
    "CD3_1:1000",
    "CD8",
    "PD-L1",
    "CK_1:150",
    "Ki67_1:150",
    "Tryptase",
    "Actin-D",
    "Caspase3-D",
    "PHH3-B",
    "Transgelin",
]

ROSIE_MARKERS = [
    "DAPI",
    "CD45",
    "CD68",
    "CD14",
    "PD1",
    "FoxP3",
    "CD8",
    "HLA-DR",
    "PanCK",
    "CD3e",
    "CD4",
    "aSMA",
    "CD31",
    "Vimentin",
    "CD45RO",
    "Ki67",
    "CD20",
    "CD11c",
    "Podoplanin",
    "PDL1",
    "GranzymeB",
    "CD38",
    "CD141",
    "CD21",
    "CD163",
    "BCL2",
    "LAG3",
    "EpCAM",
    "CD44",
    "ICOS",
    "GATA3",
    "Gal3",
    "CD39",
    "CD34",
    "TIGIT",
    "ECad",
    "CD40",
    "VISTA",
    "HLA-A",
    "MPO",
    "PCNA",
    "ATM",
    "TP63",
    "IFNg",
    "Keratin8/18",
    "IDO1",
    "CD79a",
    "HLA-E",
    "CollagenIV",
    "CD66",
]


def strip_module_prefix(state_dict):
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value
    return cleaned


def normalize_he_input(he_patch):
    he_patch = np.asarray(he_patch, dtype=np.float32)
    tensor = torch.from_numpy(he_patch).permute(2, 0, 1).unsqueeze(0)
    if tensor.max() > 1.5:
        tensor = tensor / 255.0
    return tensor


def imagenet_normalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def restore_original_he_from_cached(he_patch):
    he_patch = np.asarray(he_patch, dtype=np.float32)
    if he_patch.max() > 1.5:
        he_patch = he_patch / 255.0
    return 1.0 - he_patch


class GigaTimePredictor:
    def __init__(self, gigatime_repo, weights_path, device="cuda:0", input_size=512, batch_size=1):
        self.device = device
        self.input_size = input_size
        self.batch_size = max(1, int(batch_size))
        self.original_markers = list(GIGATIME_MARKERS)

        scripts_dir = Path(gigatime_repo) / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import archs

        self.model = archs.gigatime(len(self.original_markers), input_channels=3).to(device)
        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(strip_module_prefix(state_dict))
        self.model.eval()

    def _prepare_batch(self, he_patches):
        restored = [restore_original_he_from_cached(patch) for patch in he_patches]
        original_shapes = [(int(patch.shape[0]), int(patch.shape[1])) for patch in restored]
        tensors = []
        for patch in restored:
            tensor = normalize_he_input(patch)
            tensor = F.interpolate(tensor, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
            tensors.append(tensor)
        batch = torch.cat(tensors, dim=0).to(self.device)
        return imagenet_normalize(batch), original_shapes

    def predict_batch(self, he_patches):
        if not he_patches:
            return [], list(self.original_markers)
        batch, original_shapes = self._prepare_batch(he_patches)
        with torch.no_grad():
            pred = torch.sigmoid(self.model(batch))

        predictions = []
        for pred_one, shape_hw in zip(pred, original_shapes):
            pred_one = F.interpolate(pred_one.unsqueeze(0), size=shape_hw, mode="bilinear", align_corners=False)
            predictions.append(pred_one.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
        return predictions, list(self.original_markers)

    def predict(self, he_patch):
        preds, markers = self.predict_batch([he_patch])
        return preds[0], markers


def load_rosie_eval(rosie_repo):
    eval_path = Path(rosie_repo) / "evaluate.py"
    if not eval_path.is_file():
        raise FileNotFoundError(f"ROSIE evaluate.py not found: {eval_path}")
    spec = importlib.util.spec_from_file_location("rosie_eval", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ROSIE evaluate.py from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RosiePredictor:
    def __init__(self, rosie_repo, model_path, device="cuda:0", stride_size=8, batch_size=384, marker_order=None):
        self.device = torch.device(device)
        self.stride_size = int(stride_size)
        self.official = load_rosie_eval(rosie_repo)
        self.batch_size = int(batch_size)
        self.patch_size = int(self.official.PATCH_SIZE)
        self.original_markers = list(marker_order or ROSIE_MARKERS)
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()

        model = self.official.get_model(num_outputs=len(self.original_markers))
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if self.use_cuda:
            model = nn.DataParallel(model)
            model.load_state_dict(state)
        else:
            model.load_state_dict(strip_module_prefix(state))
        self.model = model.to(self.device)
        self.model.eval()

    def _normalize_he_rgb(self, he_patch):
        he_patch = np.asarray(he_patch, dtype=np.float32)
        if he_patch.max() <= 1.5:
            he_patch = he_patch * 255.0
        return np.clip(he_patch, 0.0, 255.0).astype(np.float32)

    def _pad_patch(self, patch, original_size, x_center, y_center):
        original_h, original_w = original_size
        current_h, current_w = patch.shape[:2]
        if current_h == self.patch_size and current_w == self.patch_size:
            return patch
        pad_left = max(self.patch_size // 2 - x_center, 0)
        pad_right = max(x_center + self.patch_size // 2 - original_w, 0)
        pad_top = max(self.patch_size // 2 - y_center, 0)
        pad_bottom = max(y_center + self.patch_size // 2 - original_h, 0)
        padded = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")
        return padded[: self.patch_size, : self.patch_size]

    def _extract_patch(self, he_patch, x_center, y_center):
        half = self.patch_size // 2
        top = int(np.clip(y_center - half, 0, he_patch.shape[0]))
        bottom = int(np.clip(y_center + half, 0, he_patch.shape[0]))
        left = int(np.clip(x_center - half, 0, he_patch.shape[1]))
        right = int(np.clip(x_center + half, 0, he_patch.shape[1]))
        patch = he_patch[top:bottom, left:right]
        return self._pad_patch(patch, he_patch.shape[:2], x_center, y_center)

    def _prepare_batch(self, patch_batch):
        tensor = torch.from_numpy(patch_batch).permute(0, 3, 1, 2).float()
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        return imagenet_normalize(tensor).to(self.device)

    def _build_coords(self, he_patch):
        h, w, _ = he_patch.shape
        stride = max(1, self.stride_size // 2)
        return [(x, y) for y in range(0, h, stride) for x in range(0, w, stride)]

    def _build_patch_batch(self, he_patch, coords):
        return np.stack([self._extract_patch(he_patch, x, y) for x, y in coords], axis=0)

    def predict(self, he_patch):
        he_patch = self._normalize_he_rgb(restore_original_he_from_cached(he_patch))
        h, w, _ = he_patch.shape
        coords = self._build_coords(he_patch)
        if not coords:
            return np.zeros((h, w, len(self.original_markers)), dtype=np.float32), list(self.original_markers)

        kernel_size = self.stride_size * 2
        yy, xx = np.mgrid[0:kernel_size, 0:kernel_size]
        center = kernel_size // 2
        weight_kernel = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * (kernel_size / 4) ** 2))

        raw_output = np.zeros((len(self.original_markers), h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        starts = range(0, len(coords), self.batch_size)
        starts = tqdm(starts, total=math.ceil(len(coords) / self.batch_size), desc="ROSIE forward", unit="batch", leave=False)
        with torch.no_grad():
            for start in starts:
                end = min(start + self.batch_size, len(coords))
                batch_coords = coords[start:end]
                batch = self._prepare_batch(self._build_patch_batch(he_patch, batch_coords))
                batch_pred = self.model(batch).detach().cpu().numpy()

                for pred_vec, (x_center, y_center) in zip(batch_pred, batch_coords):
                    half = kernel_size // 2
                    top = int(np.clip(y_center - half, 0, h))
                    bottom = int(np.clip(y_center + half, 0, h))
                    left = int(np.clip(x_center - half, 0, w))
                    right = int(np.clip(x_center + half, 0, w))
                    kh = bottom - top
                    kw = right - left
                    weight = weight_kernel[:kh, :kw]
                    raw_output[:, top:bottom, left:right] += pred_vec[:, None, None] * weight[None, :, :]
                    weight_map[top:bottom, left:right] += weight

        raw_output = raw_output / np.maximum(weight_map[None, :, :], 1e-8)
        return np.moveaxis(raw_output, 0, -1).astype(np.float32), list(self.original_markers)
