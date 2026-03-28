#!/usr/bin/env python3
"""
RIVA Dataset
=============
PyTorch Dataset and collate function for variable-length read sets.
Each sample is one locus: a (n_reads, d_read) matrix + (d_context,) vector + label.
The collate function pads read sets to the max length in the batch and creates
attention masks.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# Chromosome split definitions
SPLIT_CHROMS = {
    "train": [f"chr{i}" for i in range(1, 19)],
    "val": ["chr19"],
    "test": ["chr20"],
}


class RIVADataset(Dataset):
    """
    Loads per-read feature matrices from labeled HDF5 file.

    Each item returns:
      - reads: (n_reads, d_read) float32 tensor
      - context: (d_context,) float32 tensor
      - label: int (0 or 1)
      - meta: dict with chrom, pos, ref, alt for VCF reconstruction
    """

    def __init__(self, h5_path, split="train", chroms=None,
                 max_reads=150, exclude_unlabeled=True):
        """
        Args:
            h5_path: Path to labeled HDF5 file
            split: One of "train", "val", "test" (uses SPLIT_CHROMS)
            chroms: Override chromosome list (if None, uses split)
            max_reads: Maximum reads per locus (truncate if exceeded)
            exclude_unlabeled: Skip loci with label == -1
        """
        self.h5_path = h5_path
        self.max_reads = max_reads

        if chroms is not None:
            target_chroms = set(chroms)
        else:
            target_chroms = set(SPLIT_CHROMS.get(split, []))

        # Index all loci belonging to this split
        self.indices = []  # List of group names
        with h5py.File(h5_path, "r") as h5:
            for grp_name in sorted(h5.keys()):
                if not grp_name.startswith("locus_"):
                    continue
                grp = h5[grp_name]
                chrom = grp.attrs["chrom"]
                if chrom not in target_chroms:
                    continue
                label = int(grp.attrs.get("label", -1))
                if exclude_unlabeled and label == -1:
                    continue
                self.indices.append(grp_name)

        # We'll open the HDF5 file lazily per-worker to avoid
        # issues with multiprocessing DataLoader
        self._h5 = None

    def _get_h5(self):
        """Lazy-open HDF5 file (safe for multiprocessing DataLoader)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        h5 = self._get_h5()
        grp = h5[self.indices[idx]]

        reads = grp["reads"][:]       # (n_reads, d_read)
        context = grp["context"][:]   # (d_context,)
        label = int(grp.attrs.get("label", -1))

        # Truncate if needed
        if reads.shape[0] > self.max_reads:
            # Uniform downsample
            step = reads.shape[0] / self.max_reads
            indices = [int(i * step) for i in range(self.max_reads)]
            reads = reads[indices]

        meta = {
            "chrom": grp.attrs["chrom"],
            "pos": int(grp.attrs["pos"]),
            "ref": grp.attrs["ref"],
            "alt": grp.attrs["alt"],
            "grp_name": self.indices[idx],
        }

        return (
            torch.from_numpy(reads),       # (n, d_read)
            torch.from_numpy(context),      # (d_context,)
            torch.tensor(label, dtype=torch.float32),
            meta,
        )

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()

    def get_class_balance(self):
        """Compute class balance for loss weighting."""
        h5 = self._get_h5()
        pos = neg = 0
        for grp_name in self.indices:
            label = int(h5[grp_name].attrs.get("label", -1))
            if label == 1:
                pos += 1
            elif label == 0:
                neg += 1
        return {"pos": pos, "neg": neg, "pos_weight": neg / pos if pos > 0 else 1.0}


def riva_collate_fn(batch):
    """
    Custom collate function that pads variable-length read sets.

    Args:
        batch: list of (reads, context, label, meta) tuples

    Returns:
        reads_padded: (B, max_n, d_read) float32
        context: (B, d_context) float32
        mask: (B, max_n) bool — True for real reads, False for padding
        labels: (B,) float32
        metas: list of meta dicts
    """
    reads_list, context_list, label_list, meta_list = zip(*batch)

    # Find max reads in this batch
    max_n = max(r.shape[0] for r in reads_list)
    d_read = reads_list[0].shape[1]
    batch_size = len(reads_list)

    # Pad reads and create mask
    reads_padded = torch.zeros(batch_size, max_n, d_read)
    mask = torch.zeros(batch_size, max_n, dtype=torch.bool)

    for i, reads in enumerate(reads_list):
        n = reads.shape[0]
        reads_padded[i, :n, :] = reads
        mask[i, :n] = True

    context = torch.stack(context_list)
    labels = torch.stack(label_list)

    return reads_padded, context, mask, labels, list(meta_list)
