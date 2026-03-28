#!/usr/bin/env python3
"""
RIVA Training Script
=====================
Trains the Perceiver model on labeled HDF5 data with:
  - Chromosome-based train/val/test splits
  - BCE loss with class weighting
  - Cosine annealing LR schedule
  - Early stopping on validation loss
  - F1-optimal threshold selection on validation set
  - Checkpoint saving

Usage:
  python train.py \\
    --data labeled_features.h5 \\
    --output-dir riva_model/ \\
    --epochs 200 \\
    --batch-size 256 \\
    --hidden-dim 64 \\
    --num-latents 4
"""

import argparse
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve
)

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.perceiver import build_model
from model.dataset import RIVADataset, riva_collate_fn


def find_optimal_threshold(labels, probs, steps=200):
    """Find threshold that maximizes F1 on validation set."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    return best_threshold, best_f1


def evaluate(model, dataloader, criterion, device, threshold=0.5):
    """Evaluate model on a dataset. Returns metrics dict."""
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    n_batches = 0

    with torch.no_grad():
        for reads, context, mask, labels, metas in dataloader:
            reads = reads.to(device)
            context = context.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            logits, _ = model(reads, context, mask, mode="refine")
            logits = logits.squeeze(-1)

            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs >= threshold).astype(int)

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
        "threshold": threshold,
        "n_samples": len(all_labels),
        "n_pos": int(all_labels.sum()),
        "n_neg": int((1 - all_labels).sum()),
    }

    # AUC (only if both classes present)
    if len(np.unique(all_labels)) > 1:
        metrics["auroc"] = roc_auc_score(all_labels, all_probs)
    else:
        metrics["auroc"] = 0.0

    return metrics, all_probs, all_labels


def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    gradient_clip=1.0):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches = 0

    for reads, context, mask, labels, metas in dataloader:
        reads = reads.to(device)
        context = context.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, _ = model(reads, context, mask, mode="refine")
        logits = logits.squeeze(-1)

        loss = criterion(logits, labels)
        loss.backward()

        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="RIVA Training")
    parser.add_argument("--data", required=True,
                        help="Labeled HDF5 file from label.py")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for checkpoints and logs")

    # Model hyperparameters
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-latents", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-reads", type=int, default=150)
    parser.add_argument("--num-workers", type=int, default=4)

    # Device
    parser.add_argument("--device", default="auto",
                        help="cuda, cpu, or auto")

    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[RIVA] Device: {device}", flush=True)

    # ── Datasets ─────────────────────────────────────────────────────────
    print("[RIVA] Loading datasets...", flush=True)
    train_dataset = RIVADataset(args.data, split="train",
                                max_reads=args.max_reads)
    val_dataset = RIVADataset(args.data, split="val",
                              max_reads=args.max_reads)

    print(f"[RIVA] Train: {len(train_dataset)} loci", flush=True)
    print(f"[RIVA] Val: {len(val_dataset)} loci", flush=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=riva_collate_fn, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=riva_collate_fn, num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Class balance ────────────────────────────────────────────────────
    balance = train_dataset.get_class_balance()
    print(f"[RIVA] Class balance: {balance}", flush=True)
    pos_weight = torch.tensor([balance["pos_weight"]], dtype=torch.float32).to(device)

    # ── Model ────────────────────────────────────────────────────────────
    model_config = {
        "d_read": 18,
        "d_context": 8,
        "hidden_dim": args.hidden_dim,
        "num_latents": args.num_latents,
        "num_heads": args.num_heads,
        "num_cross_attn_layers": args.num_layers,
        "ffn_dim_multiplier": 2,
        "dropout": args.dropout,
        "latent_init": "random",
    }
    model = build_model(model_config).to(device)

    # ── Loss, optimizer, scheduler ───────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    # ── Training loop ────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_threshold = 0.5
    patience_counter = 0
    history = []

    print(f"\n[RIVA] Starting training: {args.epochs} max epochs, "
          f"patience={args.patience}", flush=True)
    print("-" * 80, flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip=args.gradient_clip
        )

        # Evaluate on validation
        val_metrics, val_probs, val_labels = evaluate(
            model, val_loader, criterion, device, threshold=best_threshold
        )

        # Find optimal threshold on validation set
        opt_threshold, opt_f1 = find_optimal_threshold(val_labels, val_probs)
        val_metrics["opt_threshold"] = opt_threshold
        val_metrics["opt_f1"] = opt_f1

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_f1": val_metrics["f1"],
            "val_opt_f1": opt_f1,
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_auroc": val_metrics["auroc"],
            "threshold": opt_threshold,
            "lr": current_lr,
            "time": elapsed,
        }
        history.append(log_entry)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"val_F1={opt_f1:.4f} | "
              f"val_AUC={val_metrics['auroc']:.4f} | "
              f"lr={current_lr:.6f} | "
              f"{elapsed:.1f}s",
              flush=True)

        # Early stopping on validation loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_f1 = opt_f1
            best_threshold = opt_threshold
            patience_counter = 0

            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model_config,
                "best_val_loss": best_val_loss,
                "best_val_f1": best_val_f1,
                "best_threshold": best_threshold,
                "class_balance": balance,
            }
            torch.save(checkpoint,
                        os.path.join(args.output_dir, "best_model.pt"))
            print(f"  → Saved best model (val_loss={best_val_loss:.4f}, "
                  f"val_F1={best_val_f1:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[RIVA] Early stopping at epoch {epoch} "
                      f"(patience={args.patience})", flush=True)
                break

    print("-" * 80)
    print(f"[RIVA] Training complete. Best val_loss={best_val_loss:.4f}, "
          f"Best val_F1={best_val_f1:.4f}, threshold={best_threshold:.4f}",
          flush=True)

    # Save training history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Final evaluation on test set ─────────────────────────────────────
    print("\n[RIVA] Evaluating on test set (chr20)...", flush=True)

    test_dataset = RIVADataset(args.data, split="test",
                               max_reads=args.max_reads)
    print(f"[RIVA] Test: {len(test_dataset)} loci", flush=True)

    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=riva_collate_fn, num_workers=args.num_workers,
        )

        # Load best model
        checkpoint = torch.load(
            os.path.join(args.output_dir, "best_model.pt"),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics, test_probs, test_labels = evaluate(
            model, test_loader, criterion, device, threshold=best_threshold
        )

        # Also find optimal test threshold (for reporting)
        test_opt_thresh, test_opt_f1 = find_optimal_threshold(
            test_labels, test_probs
        )

        print(f"\n{'='*60}")
        print(f"TEST RESULTS (chr20, threshold={best_threshold:.4f})")
        print(f"{'='*60}")
        print(f"  F1:        {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  AUROC:     {test_metrics['auroc']:.4f}")
        print(f"  Opt F1:    {test_opt_f1:.4f} (at τ={test_opt_thresh:.4f})")
        print(f"  Samples:   {test_metrics['n_samples']} "
              f"(TP={test_metrics['n_pos']}, FP={test_metrics['n_neg']})")
        print(f"{'='*60}")

        # Save test results
        test_results = {
            "metrics": test_metrics,
            "optimal_threshold": test_opt_thresh,
            "optimal_f1": test_opt_f1,
            "val_threshold_used": best_threshold,
        }
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=2)

        # Save test predictions for VCF generation
        np.savez(
            os.path.join(args.output_dir, "test_predictions.npz"),
            probs=test_probs,
            labels=test_labels,
            threshold=best_threshold,
        )

    print(f"\n[RIVA] All outputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
