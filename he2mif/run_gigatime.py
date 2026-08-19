#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path


EVA_REPO = Path(__file__).resolve().parents[1]
if str(EVA_REPO) not in sys.path:
    sys.path.insert(0, str(EVA_REPO))

from he2mif.common import (  # noqa: E402
    align_prediction,
    cached_he_from_original,
    compute_metrics,
    load_examples,
    print_metrics,
    save_comparison,
)
def add_spaformer_repo(path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args():
    parser = argparse.ArgumentParser(description="Run GigaTIME HE->MIF on Eva/examples.")
    parser.add_argument("--examples_dir", type=Path, default=EVA_REPO / "examples")
    parser.add_argument("--spaformer_repo", type=Path, default=os.environ.get("SPAFORMER_REPO"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gigatime_weights", type=Path, default=os.environ.get("GIGA_WEIGHTS"))
    parser.add_argument("--gigatime_input_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=EVA_REPO / "he2mif/results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.spaformer_repo is None:
        raise ValueError("Set --spaformer_repo or SPAFORMER_REPO.")
    add_spaformer_repo(args.spaformer_repo)
    from model_eval.he_mif_benchmark import GigaTimePredictor

    he, gt_mif, biomarkers = load_examples(args.examples_dir)
    predictor = GigaTimePredictor(
        device=args.device,
        weights_path=str(args.gigatime_weights) if args.gigatime_weights is not None else None,
        input_size=args.gigatime_input_size,
        batch_size=args.batch_size,
    )
    pred_mif, pred_markers = predictor.predict(cached_he_from_original(he))
    gt, pred, markers = align_prediction(gt_mif, biomarkers, pred_mif, pred_markers)
    pcc, ssim, rows = compute_metrics(gt, pred)
    output_path = args.output_dir / "GigaTIME_comparison.pdf"
    save_comparison(
        output_path=output_path,
        model_name="GigaTIME",
        he=he,
        gt=gt,
        pred=pred,
        markers=markers,
        pcc=pcc,
        ssim=ssim,
        rows=rows,
    )
    print_metrics("GigaTIME", output_path, pcc, ssim, markers)


if __name__ == "__main__":
    main()
