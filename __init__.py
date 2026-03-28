"""
RIVA: Read-level Indel Variant Assessment via Latent Cross-Attention

Project Structure:
==================

riva/
├── extraction/
│   ├── extract_per_read.py      # Per-read feature extraction from BAM (main script)
│   ├── features.py              # Feature computation functions per read
│   └── rescue_candidates.py     # Rescue candidate generation (Tier 1, 2, 3)
│
├── model/
│   ├── perceiver.py             # Perceiver architecture (cross-attn, self-attn, heads)
│   ├── components.py            # MHA, FFN, LayerNorm building blocks
│   └── dataset.py               # PyTorch Dataset + collate for variable-length read sets
│
├── training/
│   ├── train.py                 # Training loop with early stopping
│   ├── losses.py                # BCE, focal loss, auxiliary loss
│   └── calibrate.py             # Post-hoc isotonic calibration + ECE
│
├── evaluation/
│   ├── predict.py               # Load model, score loci, output VCF
│   ├── evaluate.py              # Run vcfeval and parse results
│   └── interpret.py             # Latent probe attention analysis
│
├── configs/
│   └── default.yaml             # Hyperparameters, paths, split definitions
│
└── utils/
    ├── vcf_utils.py             # VCF reading/writing helpers
    └── logging_utils.py         # Training logger

Pipeline:
=========
1. extract_per_read.py  →  HDF5 (per-locus read matrices + labels)
2. train.py             →  model checkpoint (.pt)
3. predict.py           →  annotated VCF
4. evaluate.py          →  vcfeval results (precision, recall, F1)
"""
