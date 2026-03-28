#!/usr/bin/env python3
"""
RIVA Labeler
=============
Annotates the extracted HDF5 feature file with ground-truth labels
from GIAB truth set comparison.

Two labeling modes:
  1. vcfeval mode: Run RTG vcfeval, parse tp/fp from output VCFs
  2. direct mode: Intersect candidate positions with truth VCF using
     a position+allele matching approach (faster, no RTG dependency)

Usage:
  python label.py \\
    --features features.h5 \\
    --truth-vcf HG002_GIAB_v4.2.1_benchmark.vcf.gz \\
    --truth-bed HG002_GIAB_v4.2.1_benchmark.bed \\
    --candidate-vcf bcftools_candidates.vcf.gz \\
    --rtg-sdf GRCh38.sdf \\
    --out labeled_features.h5 \\
    --mode vcfeval
"""

import argparse
import h5py
import os
import sys
import subprocess
import tempfile
import shutil


def parse_vcfeval_results(vcfeval_dir):
    """
    Parse vcfeval output to get TP and FP variant positions.
    Returns dict: (chrom, pos, ref, alt) -> label (1=TP, 0=FP)
    """
    from cyvcf2 import VCF

    labels = {}

    # True positives (from baseline = truth set perspective)
    tp_vcf = os.path.join(vcfeval_dir, "tp-baseline.vcf.gz")
    if os.path.exists(tp_vcf):
        for var in VCF(tp_vcf):
            if var.ALT and len(var.ALT) > 0:
                key = (var.CHROM, var.POS, var.REF, var.ALT[0])
                labels[key] = 1

    # Also check tp.vcf.gz (calls perspective) for position matching
    tp_calls_vcf = os.path.join(vcfeval_dir, "tp.vcf.gz")
    if os.path.exists(tp_calls_vcf):
        for var in VCF(tp_calls_vcf):
            if var.ALT and len(var.ALT) > 0:
                key = (var.CHROM, var.POS, var.REF, var.ALT[0])
                labels[key] = 1

    # False positives
    fp_vcf = os.path.join(vcfeval_dir, "fp.vcf.gz")
    if os.path.exists(fp_vcf):
        for var in VCF(fp_vcf):
            if var.ALT and len(var.ALT) > 0:
                key = (var.CHROM, var.POS, var.REF, var.ALT[0])
                if key not in labels:  # Don't overwrite TPs
                    labels[key] = 0

    return labels


def run_vcfeval(candidate_vcf, truth_vcf, truth_bed, rtg_sdf, output_dir,
                chroms=None):
    """Run RTG vcfeval and return the output directory."""
    cmd = [
        "rtg", "vcfeval",
        "-b", truth_vcf,
        "-c", candidate_vcf,
        "-t", rtg_sdf,
        "-o", output_dir,
        "--squash-ploidy",
    ]
    if truth_bed:
        cmd.extend(["--bed-regions", truth_bed])

    print(f"[label] Running vcfeval: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[label] vcfeval stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"vcfeval failed with return code {result.returncode}")

    print(f"[label] vcfeval output: {output_dir}", file=sys.stderr)
    return output_dir


def label_direct(features_h5_path, truth_vcf_path, truth_bed_path):
    """
    Direct labeling by intersecting positions with truth VCF.
    Simpler than vcfeval but doesn't handle representation normalization.
    Returns dict: (chrom, pos, ref, alt) -> label
    """
    from cyvcf2 import VCF

    # Load truth set positions
    truth_positions = set()
    for var in VCF(truth_vcf_path):
        if var.ALT and len(var.ALT) > 0:
            if len(var.REF) != len(var.ALT[0]):  # Indels only
                truth_positions.add((var.CHROM, var.POS))

    # Label each locus
    labels = {}
    with h5py.File(features_h5_path, "r") as h5:
        for grp_name in h5.keys():
            if not grp_name.startswith("locus_"):
                continue
            grp = h5[grp_name]
            chrom = grp.attrs["chrom"]
            pos = int(grp.attrs["pos"])
            ref_allele = grp.attrs["ref"]
            alt_allele = grp.attrs["alt"]

            key = (chrom, pos, ref_allele, alt_allele)
            if (chrom, pos) in truth_positions:
                labels[key] = 1
            else:
                labels[key] = 0

    return labels


def annotate_h5(input_path, output_path, labels):
    """Copy HDF5 file and add label attribute to each locus group."""
    labeled = 0
    unlabeled = 0

    shutil.copy2(input_path, output_path)

    with h5py.File(output_path, "a") as h5:
        for grp_name in h5.keys():
            if not grp_name.startswith("locus_"):
                continue
            grp = h5[grp_name]
            chrom = grp.attrs["chrom"]
            pos = int(grp.attrs["pos"])
            ref_allele = grp.attrs["ref"]
            alt_allele = grp.attrs["alt"]

            key = (chrom, pos, ref_allele, alt_allele)

            # Try exact match first
            if key in labels:
                grp.attrs["label"] = labels[key]
                labeled += 1
            else:
                # Try position-only match (vcfeval may normalize representation)
                pos_match = None
                for lbl_key, lbl_val in labels.items():
                    if lbl_key[0] == chrom and lbl_key[1] == pos:
                        pos_match = lbl_val
                        break
                if pos_match is not None:
                    grp.attrs["label"] = pos_match
                    labeled += 1
                else:
                    grp.attrs["label"] = -1  # Unknown / outside truth regions
                    unlabeled += 1

    print(f"[label] Labeled: {labeled}, Unlabeled: {unlabeled}",
          file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="RIVA Labeler")
    parser.add_argument("--features", required=True,
                        help="Input HDF5 from extract_per_read.py")
    parser.add_argument("--out", required=True,
                        help="Output labeled HDF5")
    parser.add_argument("--mode", choices=["vcfeval", "direct"],
                        default="vcfeval",
                        help="Labeling mode (default: vcfeval)")

    # vcfeval mode arguments
    parser.add_argument("--candidate-vcf",
                        help="Original candidate VCF (required for vcfeval mode)")
    parser.add_argument("--truth-vcf",
                        help="GIAB truth VCF")
    parser.add_argument("--truth-bed",
                        help="GIAB high-confidence regions BED")
    parser.add_argument("--rtg-sdf",
                        help="RTG SDF reference (required for vcfeval mode)")

    args = parser.parse_args()

    if args.mode == "vcfeval":
        if not all([args.candidate_vcf, args.truth_vcf, args.rtg_sdf]):
            parser.error("vcfeval mode requires --candidate-vcf, --truth-vcf, --rtg-sdf")

        vcfeval_dir = tempfile.mkdtemp(prefix="riva_vcfeval_")
        try:
            run_vcfeval(args.candidate_vcf, args.truth_vcf, args.truth_bed,
                        args.rtg_sdf, vcfeval_dir)
            labels = parse_vcfeval_results(vcfeval_dir)
        finally:
            shutil.rmtree(vcfeval_dir, ignore_errors=True)
    else:
        labels = label_direct(args.features, args.truth_vcf, args.truth_bed)

    print(f"[label] {len(labels)} labels loaded (TP: {sum(1 for v in labels.values() if v == 1)}, "
          f"FP: {sum(1 for v in labels.values() if v == 0)})",
          file=sys.stderr)

    annotate_h5(args.features, args.out, labels)
    print(f"[label] Done → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
