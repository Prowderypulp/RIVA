#!/usr/bin/env python3
"""
RIVA Per-Read Feature Extractor
================================
Refactored from DIRA v5 feature extractor. Instead of computing aggregate
statistics (means, variances, ratios) over alt/ref read groups, this script
stores the raw per-read feature vector for every read overlapping each
candidate indel locus.

Output: HDF5 file with one group per variant locus, containing:
  - reads: (n_reads, d) float32 matrix of per-read features
  - context: (d_c,) float32 vector of locus context features
  - meta: attributes (chrom, pos, ref, alt) for VCF reconstruction

Usage:
  python extract_per_read.py \\
    --vcf candidates.vcf.gz \\
    --bam sample.bam \\
    --ref GRCh38.fa \\
    --out features.h5 \\
    --chroms chr1 chr2 ... chr18 \\
    --workers 8
"""

import pysam
import math
import os
import sys
import argparse
import tempfile
import shutil
import h5py
import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MAX_READS = 150          # Truncate loci with more than this many reads
FLUSH_EVERY = 200        # Write to HDF5 every N loci

# Per-read feature names (d = 18)
READ_FEATURE_NAMES = [
    "mapping_quality",           # 0  MAPQ
    "mean_base_quality",         # 1  mean BQ across entire read
    "nbq",                       # 2  neighbourhood BQ ±5bp of variant
    "bq_at_variant",             # 3  BQ at the variant position
    "strand",                    # 4  0=forward, 1=reverse
    "proper_pair",               # 5  0/1
    "nm_edit_distance",          # 6  NM tag
    "softclip_fraction",         # 7  fraction of read that is soft-clipped
    "read_end_distance",         # 8  distance from variant to nearest read end
    "insert_size",               # 9  abs(TLEN)
    "has_indel_cigar",           # 10 read has I/D CIGAR near variant
    "cigar_indel_length",        # 11 length of CIGAR indel (0 if none)
    "cigar_matches_candidate",   # 12 CIGAR indel ≈ candidate allele length
    "local_mismatch_count",      # 13 mismatches in ±10bp window vs reference
    "mate_has_indel",            # 14 mate read also carries indel CIGAR
    "is_duplicate",              # 15 marked as duplicate
    "is_secondary",              # 16 secondary/supplementary alignment
    "alignment_score",           # 17 AS tag (0 if unavailable)
]

# Locus context feature names (d_c = 8)
CONTEXT_FEATURE_NAMES = [
    "homopolymer_length",
    "gc_content_50bp",
    "tandem_repeat_unit",
    "tandem_repeat_length",
    "indel_length",
    "is_insertion",
    "ref_entropy",
    "total_depth",
]

D_READ = len(READ_FEATURE_NAMES)    # 18
D_CONTEXT = len(CONTEXT_FEATURE_NAMES)  # 8


# ──────────────────────────────────────────────────────────────────────────────
# Pure-python helpers (from DIRA v5)
# ──────────────────────────────────────────────────────────────────────────────

def shannon_entropy(arr):
    """Shannon entropy (bits) of a sequence of discrete items."""
    n = len(arr)
    if n == 0:
        return 0.0
    counts = Counter(arr)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


# ──────────────────────────────────────────────────────────────────────────────
# NBQ with early exit (from DIRA v5)
# ──────────────────────────────────────────────────────────────────────────────

def compute_nbq(read, var_pos, window=5):
    """Neighbourhood base quality: mean BQ within ±window bp of variant."""
    quals = read.query_qualities
    if quals is None:
        return 0.0
    total = count = 0
    lo, hi = var_pos - window, var_pos + window
    for qpos, rpos in read.get_aligned_pairs(matches_only=True):
        if rpos is None:
            continue
        if rpos < lo:
            continue
        if rpos > hi:
            break
        total += quals[qpos]
        count += 1
    return (total / count) if count else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# CIGAR-aware indel support detection (from DIRA v5, unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def read_supports_indel(read, var_pos_0based, is_insertion, indel_length):
    """Check if a read's CIGAR operations support the candidate indel."""
    if read.cigartuples is None:
        return False

    tolerance = 2
    ref_pos = read.reference_start
    target_op = 1 if is_insertion else 2

    for op, length in read.cigartuples:
        if op == 0 or op == 7 or op == 8:  # M, =, X
            ref_pos += length
        elif op == 1:  # I
            if op == target_op:
                if abs(ref_pos - var_pos_0based) <= tolerance:
                    if abs(length - abs(indel_length)) <= max(1, abs(indel_length) // 2):
                        return True
        elif op == 2:  # D
            if op == target_op:
                del_start = ref_pos
                del_end = ref_pos + length
                if del_start - tolerance <= var_pos_0based <= del_end + tolerance:
                    if abs(length - abs(indel_length)) <= max(1, abs(indel_length) // 2):
                        return True
            ref_pos += length
        elif op == 3:  # N
            ref_pos += length
        elif op == 4 or op == 5:  # S, H
            pass

    return False


def read_has_any_indel_near(read, var_pos_0based, window=2):
    """
    Check if a read has ANY indel CIGAR operation near the position,
    regardless of candidate allele. Returns (has_indel, indel_length).
    Used for per-read feature encoding without assuming a specific candidate.
    """
    if read.cigartuples is None:
        return False, 0

    ref_pos = read.reference_start
    for op, length in read.cigartuples:
        if op == 0 or op == 7 or op == 8:  # M, =, X
            ref_pos += length
        elif op == 1:  # Insertion
            if abs(ref_pos - var_pos_0based) <= window:
                return True, length
        elif op == 2:  # Deletion
            del_start = ref_pos
            del_end = ref_pos + length
            if del_start - window <= var_pos_0based <= del_end + window:
                return True, length
            ref_pos += length
        elif op == 3:  # N
            ref_pos += length
        elif op == 4 or op == 5:  # S, H
            pass

    return False, 0


# ──────────────────────────────────────────────────────────────────────────────
# Tandem repeat detection (from DIRA v5, unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def find_tandem_repeat_length(seq, pos_in_seq, max_unit=6):
    """Find the longest tandem repeat spanning pos_in_seq."""
    if not seq or pos_in_seq < 0 or pos_in_seq >= len(seq):
        return 1, 1

    best_total = 1
    best_unit = 1

    for unit_len in range(1, max_unit + 1):
        if pos_in_seq + unit_len > len(seq):
            break
        unit = seq[pos_in_seq:pos_in_seq + unit_len]

        left = pos_in_seq
        while left >= unit_len and seq[left - unit_len:left] == unit:
            left -= unit_len

        right = pos_in_seq + unit_len
        while right + unit_len <= len(seq) and seq[right:right + unit_len] == unit:
            right += unit_len

        total = right - left
        if total > best_total:
            best_total = total
            best_unit = unit_len

    return best_unit, best_total


# ──────────────────────────────────────────────────────────────────────────────
# Per-read mismatch counting against reference (from DIRA v5 flanking_bq)
# ──────────────────────────────────────────────────────────────────────────────

def count_local_mismatches(read, var_pos, ref_obj, chrom, window=10):
    """Count actual mismatches vs reference in ±window bp around variant."""
    quals = read.query_qualities
    query = read.query_sequence
    if quals is None or query is None:
        return 0

    lo, hi = var_pos - window, var_pos + window
    mismatches = 0

    try:
        ref_seq = ref_obj.fetch(chrom, max(0, lo), hi + 1)
        ref_offset = max(0, lo)
    except Exception:
        return 0

    for qpos, rpos in read.get_aligned_pairs(matches_only=True):
        if rpos is None:
            continue
        if rpos < lo:
            continue
        if rpos > hi:
            break
        ref_idx = rpos - ref_offset
        if 0 <= ref_idx < len(ref_seq):
            if query[qpos].upper() != ref_seq[ref_idx].upper():
                mismatches += 1

    return mismatches


# ──────────────────────────────────────────────────────────────────────────────
# Extract per-read feature vector for a single read at a locus
# ──────────────────────────────────────────────────────────────────────────────

def extract_read_features(read, var_pos_0based, var_pos_1based, is_insertion,
                          indel_length, ref_obj, chrom, alt_read_names):
    """
    Extract a d-dimensional feature vector for one read at one locus.
    Returns a numpy array of shape (D_READ,).
    """
    feat = np.zeros(D_READ, dtype=np.float32)

    # 0: Mapping quality
    feat[0] = float(read.mapping_quality)

    # 1: Mean base quality across entire read
    quals = read.query_qualities
    if quals is not None and len(quals) > 0:
        feat[1] = sum(quals) / len(quals)

    # 2: Neighbourhood base quality (±5bp of variant)
    feat[2] = compute_nbq(read, var_pos_1based, window=5)

    # 3: Base quality at variant position
    for qpos, rpos in read.get_aligned_pairs():
        if rpos == var_pos_0based and qpos is not None:
            if quals is not None and qpos < len(quals):
                feat[3] = float(quals[qpos])
            break

    # 4: Strand (0=forward, 1=reverse)
    feat[4] = 1.0 if read.is_reverse else 0.0

    # 5: Proper pair
    feat[5] = 1.0 if read.is_proper_pair else 0.0

    # 6: NM edit distance
    try:
        feat[6] = float(read.get_tag("NM"))
    except KeyError:
        feat[6] = 0.0

    # 7: Soft-clip fraction
    ct = read.cigartuples
    if ct:
        sc_total = sum(length for op, length in ct if op == 4)
        rlen = read.query_length or read.infer_read_length() or 150
        feat[7] = sc_total / rlen if rlen > 0 else 0.0

    # 8: Read-end distance (distance from variant to nearest read end)
    rlen = read.query_length or read.infer_read_length() or 150
    qpos_at_site = None
    for qp, rp in read.get_aligned_pairs():
        if rp == var_pos_0based:
            qpos_at_site = qp
            break
    if qpos_at_site is not None:
        feat[8] = float(min(qpos_at_site, rlen - qpos_at_site))
    else:
        feat[8] = float(rlen // 2)

    # 9: Insert size (absolute TLEN)
    feat[9] = float(abs(read.template_length)) if read.template_length != 0 else 0.0

    # 10: Has indel CIGAR near variant
    has_indel, cigar_indel_len = read_has_any_indel_near(read, var_pos_0based, window=2)
    feat[10] = 1.0 if has_indel else 0.0

    # 11: CIGAR indel length
    feat[11] = float(cigar_indel_len)

    # 12: CIGAR matches candidate (length validation from DIRA)
    if has_indel and indel_length != 0:
        length_tol = max(1, abs(indel_length) // 2)
        feat[12] = 1.0 if abs(cigar_indel_len - abs(indel_length)) <= length_tol else 0.0
    else:
        feat[12] = 0.0

    # 13: Local mismatch count (±10bp vs reference)
    feat[13] = float(count_local_mismatches(read, var_pos_0based, ref_obj, chrom, window=10))

    # 14: Mate has indel (mate's query_name is in alt_read_names)
    if read.is_paired and not read.mate_is_unmapped:
        feat[14] = 1.0 if read.query_name in alt_read_names else 0.0
    else:
        feat[14] = 0.0

    # 15: Is duplicate
    feat[15] = 1.0 if read.is_duplicate else 0.0

    # 16: Is secondary/supplementary
    feat[16] = 1.0 if (read.is_secondary or read.is_supplementary) else 0.0

    # 17: Alignment score (AS tag)
    try:
        feat[17] = float(read.get_tag("AS"))
    except KeyError:
        feat[17] = 0.0

    return feat


# ──────────────────────────────────────────────────────────────────────────────
# Extract locus context vector
# ──────────────────────────────────────────────────────────────────────────────

def extract_locus_context(var_pos_0based, indel_length, is_insertion, depth,
                          ref_obj, chrom):
    """
    Extract locus context features shared across all reads.
    Returns a numpy array of shape (D_CONTEXT,).
    """
    ctx = np.zeros(D_CONTEXT, dtype=np.float32)

    # Fetch reference context ±50bp
    ctx_start = max(0, var_pos_0based - 50)
    ctx_end = var_pos_0based + 50
    try:
        seq = ref_obj.fetch(chrom, ctx_start, ctx_end)
    except Exception:
        seq = ""

    if len(seq) == 0:
        ctx[0] = 1.0   # homopolymer_length
        ctx[1] = 0.0   # gc_content
        ctx[2] = 1.0   # tandem_repeat_unit
        ctx[3] = 1.0   # tandem_repeat_length
    else:
        # GC content
        ctx[1] = (seq.count("G") + seq.count("C")) / len(seq)

        ci = var_pos_0based - ctx_start
        if ci < 0 or ci >= len(seq):
            ctx[0] = 1.0
            ctx[2] = 1.0
            ctx[3] = 1.0
        else:
            # Homopolymer run length
            center = seq[ci]
            homopoly = 1
            j = ci - 1
            while j >= 0 and seq[j] == center:
                homopoly += 1
                j -= 1
            j = ci + 1
            while j < len(seq) and seq[j] == center:
                homopoly += 1
                j += 1
            ctx[0] = float(homopoly)

            # Tandem repeat
            tr_unit, tr_length = find_tandem_repeat_length(seq, ci)
            ctx[2] = float(tr_unit)
            ctx[3] = float(tr_length)

    # Indel length and direction
    ctx[4] = float(abs(indel_length))
    ctx[5] = float(is_insertion)

    # Reference entropy in ±10bp window
    ref_start = max(0, var_pos_0based - 10)
    ref_end = var_pos_0based + 10
    try:
        ref_local = ref_obj.fetch(chrom, ref_start, ref_end)
        ctx[6] = shannon_entropy(list(ref_local.upper()))
    except Exception:
        ctx[6] = 0.0

    # Total depth
    ctx[7] = float(depth)

    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# Worker: process all indels on one chromosome → temp HDF5
# ──────────────────────────────────────────────────────────────────────────────

def process_chromosome(chrom, vcf_path, bam_path, ref_path, tmp_dir):
    """
    Process one chromosome: iterate over candidate indels, extract per-read
    features, write to a temporary HDF5 file.
    """
    from cyvcf2 import VCF

    vcf = VCF(vcf_path)
    bam = pysam.AlignmentFile(bam_path, "rb")
    ref = pysam.FastaFile(ref_path)

    tmp_path = os.path.join(tmp_dir, f"{chrom}.h5")
    h5 = h5py.File(tmp_path, "w")
    locus_idx = 0

    try:
        var_iter = vcf(chrom)
    except Exception:
        bam.close()
        ref.close()
        h5.close()
        return tmp_path

    for var in var_iter:
        # Skip non-indels
        if var.ALT is None or len(var.ALT) == 0:
            continue
        if len(var.REF) == len(var.ALT[0]):
            continue

        pos = var.POS          # 1-based
        alt = var.ALT[0]
        ref_allele = var.REF
        pos0 = pos - 1         # 0-based

        indel_length = len(alt) - len(ref_allele)
        is_insertion = 1 if indel_length > 0 else 0

        # ── Collect reads via fetch ──────────────────────────────────────
        all_reads = []
        alt_read_names = set()

        for read in bam.fetch(chrom, max(0, pos0 - 1), pos0 + 2):
            if read.is_unmapped:
                continue
            # Keep secondary/supplementary — mark them in features instead
            # of filtering them out. Let the model learn what to do.
            if read.is_duplicate:
                continue
            if read.reference_start > pos0 or read.reference_end is None:
                continue
            if read.reference_end <= pos0:
                continue

            all_reads.append(read)

        depth = len(all_reads)
        if depth == 0:
            continue

        # ── First pass: identify alt read names for mate feature ──────────
        for read in all_reads:
            if read.is_secondary or read.is_supplementary:
                continue
            if read_supports_indel(read, pos0, is_insertion == 1, indel_length):
                alt_read_names.add(read.query_name)

        # ── Truncate if too many reads (downsample uniformly) ─────────────
        if depth > MAX_READS:
            step = depth / MAX_READS
            indices = [int(i * step) for i in range(MAX_READS)]
            all_reads = [all_reads[i] for i in indices]
            depth = MAX_READS

        # ── Extract per-read feature matrix ───────────────────────────────
        read_matrix = np.zeros((depth, D_READ), dtype=np.float32)

        for i, read in enumerate(all_reads):
            read_matrix[i] = extract_read_features(
                read, pos0, pos, is_insertion, indel_length,
                ref, chrom, alt_read_names
            )

        # ── Extract locus context ─────────────────────────────────────────
        context_vec = extract_locus_context(
            pos0, indel_length, is_insertion, depth, ref, chrom
        )

        # ── Write to HDF5 ────────────────────────────────────────────────
        grp_name = f"locus_{locus_idx:07d}"
        grp = h5.create_group(grp_name)
        grp.create_dataset("reads", data=read_matrix, compression="gzip",
                           compression_opts=4)
        grp.create_dataset("context", data=context_vec)

        # Store metadata as attributes for VCF reconstruction
        grp.attrs["chrom"] = chrom
        grp.attrs["pos"] = pos          # 1-based
        grp.attrs["ref"] = ref_allele
        grp.attrs["alt"] = alt
        grp.attrs["indel_length"] = indel_length
        grp.attrs["n_reads"] = depth

        locus_idx += 1

    h5.close()
    bam.close()
    ref.close()

    print(f"  [{chrom}] {locus_idx} loci extracted → {tmp_path}",
          file=sys.stderr, flush=True)
    return tmp_path


# ──────────────────────────────────────────────────────────────────────────────
# Merge chromosome HDF5 files into one
# ──────────────────────────────────────────────────────────────────────────────

def merge_h5(tmp_dir, chroms, out_path):
    """Merge per-chromosome HDF5 files into a single output file."""
    with h5py.File(out_path, "w") as out_h5:
        # Store feature name metadata
        out_h5.attrs["read_feature_names"] = READ_FEATURE_NAMES
        out_h5.attrs["context_feature_names"] = CONTEXT_FEATURE_NAMES
        out_h5.attrs["d_read"] = D_READ
        out_h5.attrs["d_context"] = D_CONTEXT

        global_idx = 0
        for chrom in chroms:
            tmp_path = os.path.join(tmp_dir, f"{chrom}.h5")
            if not os.path.exists(tmp_path):
                continue

            with h5py.File(tmp_path, "r") as src:
                for grp_name in sorted(src.keys()):
                    src_grp = src[grp_name]
                    dst_name = f"locus_{global_idx:07d}"

                    dst_grp = out_h5.create_group(dst_name)
                    dst_grp.create_dataset("reads",
                                           data=src_grp["reads"][:],
                                           compression="gzip",
                                           compression_opts=4)
                    dst_grp.create_dataset("context",
                                           data=src_grp["context"][:])

                    # Copy attributes
                    for key, val in src_grp.attrs.items():
                        dst_grp.attrs[key] = val

                    global_idx += 1

        out_h5.attrs["n_loci"] = global_idx
        print(f"[merge] {global_idx} total loci → {out_path}",
              file=sys.stderr, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Worker wrapper for multiprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _worker_wrapper(chrom, vcf_path, bam_path, ref_path, tmp_dir):
    print(f"  [{chrom}] started", file=sys.stderr, flush=True)
    result = process_chromosome(chrom, vcf_path, bam_path, ref_path, tmp_dir)
    print(f"  [{chrom}] done", file=sys.stderr, flush=True)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RIVA Per-Read Feature Extractor")
    parser.add_argument("--vcf", required=True,
                        help="Candidate VCF (bcftools, FreeBayes, etc.)")
    parser.add_argument("--bam", required=True,
                        help="Aligned BAM file")
    parser.add_argument("--ref", required=True,
                        help="Reference FASTA (must be indexed)")
    parser.add_argument("--out", required=True,
                        help="Output HDF5 file")
    parser.add_argument("--chroms", nargs="+",
                        default=[f"chr{i}" for i in range(1, 23)],
                        help="Chromosomes to process (default: chr1-chr22)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: auto)")
    parser.add_argument("--max-reads", type=int, default=150,
                        help="Max reads per locus (default: 150)")
    args = parser.parse_args()

    global MAX_READS
    MAX_READS = args.max_reads

    chroms = args.chroms
    n_workers = args.workers if args.workers > 0 else min(len(chroms), cpu_count())

    print(f"[RIVA extract] {len(chroms)} chromosomes, "
          f"{n_workers} workers, max_reads={MAX_READS}",
          file=sys.stderr)

    tmp_dir = tempfile.mkdtemp(prefix="riva_extract_")
    try:
        worker_fn = partial(
            _worker_wrapper,
            vcf_path=args.vcf,
            bam_path=args.bam,
            ref_path=args.ref,
            tmp_dir=tmp_dir,
        )
        with Pool(processes=n_workers) as pool:
            pool.map(worker_fn, chroms)

        merge_h5(tmp_dir, chroms, args.out)
        print(f"[RIVA extract] Done → {args.out}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
