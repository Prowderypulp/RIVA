#!/usr/bin/env python3
"""
RIVA Fast Labeler
==================
Runs vcfeval, builds a position lookup, and annotates the existing
HDF5 file IN-PLACE (no copying). Much faster than the original.

Usage:
  python3 label_fast.py \
    --features /data2/riva/features.h5 \
    --candidate-vcf /data2/riva/vcf/HG002_bcftools.vcf.gz \
    --truth-vcf /data2/riva/truth/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz \
    --truth-bed /data2/riva/truth/HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed \
    --rtg-sdf /data2/riva/ref/GRCh38.sdf
"""

import argparse
import subprocess
import tempfile
import shutil
import sys
import os
import time
import h5py
from cyvcf2 import VCF


def run_vcfeval(candidate_vcf, truth_vcf, truth_bed, rtg_sdf, output_dir):
    """Run RTG vcfeval."""
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

    print(f"[label] Running: {' '.join(cmd)}", file=sys.stderr, flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[label] STDERR: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"vcfeval failed (rc={result.returncode})")

    # Print vcfeval summary
    summary = os.path.join(output_dir, "summary.txt")
    if os.path.exists(summary):
        with open(summary) as f:
            print(f.read(), file=sys.stderr)

    return output_dir


def build_label_lookup(vcfeval_dir):
    """
    Build a fast lookup dict from vcfeval TP and FP VCFs.
    Key: (chrom, pos) -> label (1=TP, 0=FP)
    Uses position-only matching for speed. If a position appears
    in both TP and FP (different alleles), TP wins.
    """
    labels = {}

    # True positives from calls perspective
    tp_vcf = os.path.join(vcfeval_dir, "tp.vcf.gz")
    if os.path.exists(tp_vcf):
        tp_count = 0
        for var in VCF(tp_vcf):
            if var.ALT and len(var.ALT) > 0:
                if len(var.REF) != len(var.ALT[0]):  # indels only
                    labels[(var.CHROM, var.POS)] = 1
                    tp_count += 1
        print(f"[label] TP indels from vcfeval: {tp_count}", file=sys.stderr, flush=True)

    # False positives
    fp_vcf = os.path.join(vcfeval_dir, "fp.vcf.gz")
    if os.path.exists(fp_vcf):
        fp_count = 0
        for var in VCF(fp_vcf):
            if var.ALT and len(var.ALT) > 0:
                if len(var.REF) != len(var.ALT[0]):  # indels only
                    key = (var.CHROM, var.POS)
                    if key not in labels:  # don't overwrite TPs
                        labels[key] = 0
                        fp_count += 1
        print(f"[label] FP indels from vcfeval: {fp_count}", file=sys.stderr, flush=True)

    return labels


def annotate_h5_inplace(h5_path, labels):
    """Annotate HDF5 in-place by adding label attribute to each group."""
    t0 = time.time()
    labeled = 0
    unlabeled = 0
    tp = 0
    fp = 0

    with h5py.File(h5_path, "a") as h5:
        keys = [k for k in h5.keys() if k.startswith("locus_")]
        total = len(keys)
        print(f"[label] Annotating {total} loci...", file=sys.stderr, flush=True)

        for i, grp_name in enumerate(keys):
            grp = h5[grp_name]
            chrom = grp.attrs["chrom"]
            pos = int(grp.attrs["pos"])

            key = (chrom, pos)
            if key in labels:
                lbl = labels[key]
                grp.attrs["label"] = lbl
                labeled += 1
                if lbl == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                grp.attrs["label"] = -1
                unlabeled += 1

            if (i + 1) % 100000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                print(f"  {i+1}/{total} ({rate:.0f} loci/s, ETA {eta:.0f}s) "
                      f"TP={tp} FP={fp} unlabeled={unlabeled}",
                      file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    print(f"[label] Done in {elapsed:.1f}s", file=sys.stderr, flush=True)
    print(f"[label] TP={tp}, FP={fp}, unlabeled={unlabeled}, total={labeled+unlabeled}",
          file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description="RIVA Fast Labeler")
    parser.add_argument("--features", required=True)
    parser.add_argument("--candidate-vcf", required=True)
    parser.add_argument("--truth-vcf", required=True)
    parser.add_argument("--truth-bed", default=None)
    parser.add_argument("--rtg-sdf", required=True)
    parser.add_argument("--vcfeval-dir", default=None,
                        help="Reuse existing vcfeval output (skip rerun)")
    args = parser.parse_args()

    # Step 1: Run vcfeval (or reuse existing)
    if args.vcfeval_dir and os.path.exists(args.vcfeval_dir):
        print(f"[label] Reusing vcfeval output: {args.vcfeval_dir}",
              file=sys.stderr, flush=True)
        vcfeval_dir = args.vcfeval_dir
        cleanup = False
    else:
        vcfeval_dir = tempfile.mkdtemp(prefix="riva_vcfeval_")
        cleanup = True
        run_vcfeval(args.candidate_vcf, args.truth_vcf, args.truth_bed,
                    args.rtg_sdf, vcfeval_dir)

    # Step 2: Build lookup
    print("[label] Building label lookup...", file=sys.stderr, flush=True)
    labels = build_label_lookup(vcfeval_dir)
    print(f"[label] {len(labels)} labeled positions loaded",
          file=sys.stderr, flush=True)

    # Step 3: Annotate in-place
    annotate_h5_inplace(args.features, labels)

    # Cleanup
    if cleanup:
        shutil.rmtree(vcfeval_dir, ignore_errors=True)

    print("[label] Complete.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
