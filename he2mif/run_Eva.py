#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

EVA_REPO = Path(__file__).resolve().parents[1]
if str(EVA_REPO) not in sys.path:
    sys.path.insert(0, str(EVA_REPO))

from Eva.utils import load_from_hf  # noqa: E402
from he2mif.common import align_prediction, compute_metrics, load_examples, print_metrics, save_comparison  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run Eva_ft HE->MIF on Eva/examples.")
    parser.add_argument("--examples_dir", type=Path, default=EVA_REPO / "examples")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repo_id", default="yandrewl/Eva")
    parser.add_argument("--checkpoint_filename", default="Eva_ft.ckpt")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=EVA_REPO / "he2mif/results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    he, gt_mif, biomarkers = load_examples(args.examples_dir)

    conf = OmegaConf.load(EVA_REPO / "config.yaml")
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    model = load_from_hf(
        repo_id=args.repo_id,
        conf=conf,
        device=device,
        checkpoint_filename=args.checkpoint_filename,
    )
    model.eval()

    model_input = np.concatenate([np.zeros_like(gt_mif), 1.0 - he], axis=-1).astype(np.float32)
    marker_in = biomarkers + ["HECHA1", "HECHA2", "HECHA3"]
    marker_out = biomarkers
    num_tokens = (he.shape[0] // model.token_size) * (he.shape[1] // model.token_size)
    infer_mask = torch.zeros(len(marker_in), num_tokens, device=device)
    infer_mask[: len(biomarkers), :] = 1.0

    with torch.no_grad():
        batch = torch.from_numpy(model_input[None]).permute(0, 3, 1, 2).contiguous().float().to(device)
        image_recon_cls, _ = model.model.forward(
            imgs=batch,
            marker_in=[marker_in],
            marker_out=[marker_out],
            infer_mask=infer_mask,
            channel_mask=None,
        )
        recon_tokens = image_recon_cls[:, :, 1:, :]
        pred_mif = rearrange(
            recon_tokens,
            "N C (H W) (P1 P2) -> N (H P1) (W P2) C",
            P1=model.token_size,
            P2=model.token_size,
            H=model.img_size // model.token_size,
            N=image_recon_cls.shape[0],
        )[0].detach().cpu().numpy()

    gt, pred, markers = align_prediction(gt_mif, biomarkers, pred_mif, biomarkers)
    pcc, ssim, rows = compute_metrics(gt, pred)
    output_path = args.output_dir / "Eva_comparison.pdf"
    save_comparison(
        output_path=output_path,
        model_name="Eva_ft",
        he=he,
        gt=gt,
        pred=pred,
        markers=markers,
        pcc=pcc,
        ssim=ssim,
        rows=rows,
    )
    print_metrics("Eva_ft", output_path, pcc, ssim, markers)


if __name__ == "__main__":
    main()
