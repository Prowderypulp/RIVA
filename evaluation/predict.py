#!/usr/bin/env python3
"""
RIVA Predict
=============
Load a trained RIVA model, score candidate loci, and output an annotated VCF.

Usage:
  python predict.py \\
    --data features.h5 \\
    --model riva_model/best_model.pt \\
    --input-vcf bcftools_candidates.vcf.gz \\
    --output-vcf riva_filtered.vcf.gz \\
    --chroms chr20
"""

import argparse
import os
import sys
import gzip
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.perceiver import build_model
from model.dataset import RIVADataset, riva_collate_fn
from torch.utils.data import DataLoader


def phred_scale(prob):
    """Convert probability to Phred-scaled quality score."""
    if prob >= 1.0:
        return 999.0
    if prob <= 0.0:
        return 0.0
    return -10.0 * np.log10(1.0 - prob)


def predict_all(model, dataloader, device, threshold=0.5):
    """
    Run prediction on all loci in dataloader.
    Returns list of (meta_dict, probability, pass/fail) tuples.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for reads, context, mask, labels, metas in dataloader:
            reads = reads.to(device)
            context = context.to(device)
            mask = mask.to(device)

            logits, _ = model(reads, context, mask, mode="refine")
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

            for i, prob in enumerate(probs):
                results.append({
                    "chrom": metas[i]["chrom"],
                    "pos": metas[i]["pos"],
                    "ref": metas[i]["ref"],
                    "alt": metas[i]["alt"],
                    "prob": float(prob),
                    "qual": float(phred_scale(prob)),
                    "pass": bool(prob >= threshold),
                })

    return results


def annotate_vcf(input_vcf, output_vcf, predictions, threshold):
    """
    Read input VCF, annotate indels with RIVA scores, write output VCF.
    Non-indel records are passed through unchanged.
    """
    # Build lookup: (chrom, pos, ref, alt) -> prediction
    pred_lookup = {}
    for pred in predictions:
        key = (pred["chrom"], pred["pos"], pred["ref"], pred["alt"])
        pred_lookup[key] = pred

    # Determine open function based on extension
    if input_vcf.endswith(".gz"):
        open_in = lambda: gzip.open(input_vcf, "rt")
    else:
        open_in = lambda: open(input_vcf, "r")

    if output_vcf.endswith(".gz"):
        open_out = lambda: gzip.open(output_vcf, "wt")
    else:
        open_out = lambda: open(output_vcf, "w")

    n_annotated = 0
    n_filtered = 0
    n_passed = 0
    n_total = 0

    with open_in() as fin, open_out() as fout:
        for line in fin:
            if line.startswith("##"):
                fout.write(line)
                # Add RIVA header lines before first non-header line
                continue
            elif line.startswith("#CHROM"):
                # Add RIVA-specific header lines
                fout.write('##FILTER=<ID=RIVA_FAIL,Description="Failed RIVA '
                           f'refinement (threshold={threshold:.4f})">\n')
                fout.write('##INFO=<ID=RIVA_PROB,Number=1,Type=Float,'
                           'Description="RIVA probability of true variant">\n')
                fout.write('##INFO=<ID=RIVA_QUAL,Number=1,Type=Float,'
                           'Description="RIVA Phred-scaled quality score">\n')
                fout.write(line)
                continue

            n_total += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                fout.write(line)
                continue

            chrom = fields[0]
            pos = int(fields[1])
            ref_allele = fields[3]
            alt_allele = fields[4]

            # Check if this is an indel
            is_indel = len(ref_allele) != len(alt_allele)

            if not is_indel:
                fout.write(line)
                continue

            key = (chrom, pos, ref_allele, alt_allele)
            if key in pred_lookup:
                pred = pred_lookup[key]
                n_annotated += 1

                # Update QUAL
                fields[5] = f"{pred['qual']:.1f}"

                # Update FILTER
                if pred["pass"]:
                    fields[6] = "PASS"
                    n_passed += 1
                else:
                    fields[6] = "RIVA_FAIL"
                    n_filtered += 1

                # Add INFO annotations
                info = fields[7]
                info += f";RIVA_PROB={pred['prob']:.4f}"
                info += f";RIVA_QUAL={pred['qual']:.1f}"
                fields[7] = info

            fout.write("\t".join(fields) + "\n")

    print(f"[predict] VCF annotation complete:", file=sys.stderr)
    print(f"  Total records: {n_total}", file=sys.stderr)
    print(f"  Annotated (indels): {n_annotated}", file=sys.stderr)
    print(f"  PASS: {n_passed}", file=sys.stderr)
    print(f"  RIVA_FAIL: {n_filtered}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="RIVA Prediction")
    parser.add_argument("--data", required=True,
                        help="HDF5 features file")
    parser.add_argument("--model", required=True,
                        help="Trained model checkpoint (.pt)")
    parser.add_argument("--input-vcf", required=True,
                        help="Original candidate VCF")
    parser.add_argument("--output-vcf", required=True,
                        help="Output annotated VCF")
    parser.add_argument("--chroms", nargs="+", default=["chr20"],
                        help="Chromosomes to predict on")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load model
    print(f"[predict] Loading model from {args.model}", file=sys.stderr)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    threshold = checkpoint["best_threshold"]

    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[predict] Threshold: {threshold:.4f}", file=sys.stderr)

    # Load dataset for target chromosomes
    dataset = RIVADataset(
        args.data, chroms=args.chroms,
        max_reads=150, exclude_unlabeled=False,
    )
    print(f"[predict] {len(dataset)} loci to score", file=sys.stderr)

    if len(dataset) == 0:
        print("[predict] No loci found, exiting.", file=sys.stderr)
        return

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=riva_collate_fn, num_workers=4,
    )

    # Predict
    predictions = predict_all(model, dataloader, device, threshold=threshold)
    print(f"[predict] {len(predictions)} predictions generated",
          file=sys.stderr)

    # Annotate VCF
    annotate_vcf(args.input_vcf, args.output_vcf, predictions, threshold)
    print(f"[predict] Done → {args.output_vcf}", file=sys.stderr)


if __name__ == "__main__":
    main()
