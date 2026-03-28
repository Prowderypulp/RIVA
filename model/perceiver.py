#!/usr/bin/env python3
"""
RIVA Perceiver Model
=====================
Perceiver-style latent cross-attention architecture for read-level
indel variant assessment.

Architecture:
  1. Read Encoder MLP: per-read features → h-dim embeddings
  2. Context Encoder: locus context → h-dim token
  3. Latent Cross-Attention: K latent probes query read set
  4. Latent Self-Attention: latents interact
  5. Mean Pool → Classification Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional masking."""

    def __init__(self, h_dim, num_heads, dropout=0.1):
        super().__init__()
        assert h_dim % num_heads == 0
        self.h_dim = h_dim
        self.num_heads = num_heads
        self.d_k = h_dim // num_heads

        self.W_q = nn.Linear(h_dim, h_dim)
        self.W_k = nn.Linear(h_dim, h_dim)
        self.W_v = nn.Linear(h_dim, h_dim)
        self.W_o = nn.Linear(h_dim, h_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (B, m, h)  — queries
            K: (B, n, h)  — keys
            V: (B, n, h)  — values
            mask: (B, n) bool — True for real tokens, False for padding
                  Applied to key/value positions

        Returns:
            output: (B, m, h)
            attn_weights: (B, num_heads, m, n) — for interpretability
        """
        B, m, _ = Q.shape
        n = K.shape[1]

        # Project and reshape to (B, num_heads, seq_len, d_k)
        q = self.W_q(Q).view(B, m, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, n, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (B, num_heads, m, n)

        if mask is not None:
            # mask: (B, n) → (B, 1, 1, n) for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, n)
            scores = scores.masked_fill(~mask_expanded, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Handle NaN from all-masked positions
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        out = torch.matmul(attn_weights, v)  # (B, num_heads, m, d_k)
        out = out.transpose(1, 2).contiguous().view(B, m, self.h_dim)
        out = self.W_o(out)

        return out, attn_weights


class FeedForward(nn.Module):
    """Two-layer FFN with GELU activation."""

    def __init__(self, h_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, h_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    Latent cross-attention: latent probes (Q) attend to read tokens (KV).
    Followed by FFN with residual connections and layer norm.
    """

    def __init__(self, h_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttention(h_dim, num_heads, dropout)
        self.ffn = FeedForward(h_dim, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, latents, tokens, mask=None):
        """
        Args:
            latents: (B, K, h) — latent probes
            tokens: (B, n+1, h) — read embeddings + context token
            mask: (B, n+1) — padding mask for tokens

        Returns:
            latents: (B, K, h) — updated latent probes
            attn_weights: (B, num_heads, K, n+1) — cross-attention weights
        """
        # Cross-attention with residual
        attn_out, attn_weights = self.cross_attn(
            Q=latents, K=tokens, V=tokens, mask=mask
        )
        latents = self.ln1(latents + attn_out)

        # FFN with residual
        latents = self.ln2(latents + self.ffn(latents))

        return latents, attn_weights


class SelfAttentionBlock(nn.Module):
    """
    Latent self-attention: latent probes attend to each other.
    No masking needed since all K latents are always present.
    """

    def __init__(self, h_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(h_dim, num_heads, dropout)
        self.ffn = FeedForward(h_dim, ff_dim, dropout)
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, latents):
        """
        Args:
            latents: (B, K, h)
        Returns:
            latents: (B, K, h)
        """
        attn_out, _ = self.self_attn(Q=latents, K=latents, V=latents)
        latents = self.ln1(latents + attn_out)
        latents = self.ln2(latents + self.ffn(latents))
        return latents


class RIVAPerceiver(nn.Module):
    """
    Full RIVA Perceiver model.

    Components:
      - Read encoder: MLP projecting per-read features to h_dim
      - Context encoder: Linear projecting locus context to h_dim
      - Latent probes: K learned vectors in R^h
      - Cross-attention blocks: latents query read tokens
      - Self-attention blocks: latents interact
      - Classification head: mean-pool latents → MLP → sigmoid
    """

    def __init__(
        self,
        d_read=18,
        d_context=8,
        h_dim=64,
        num_latents=4,
        num_heads=4,
        num_layers=1,       # Number of cross-attn + self-attn rounds
        ff_multiplier=2,
        dropout=0.1,
        latent_init="random",  # "random" or "domain"
    ):
        super().__init__()

        self.h_dim = h_dim
        self.num_latents = num_latents
        self.num_layers = num_layers

        # ── Read Encoder ─────────────────────────────────────────────────
        self.read_encoder = nn.Sequential(
            nn.Linear(d_read, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
        )

        # ── Context Encoder ──────────────────────────────────────────────
        self.context_encoder = nn.Sequential(
            nn.Linear(d_context, h_dim),
            nn.GELU(),
            nn.LayerNorm(h_dim),
        )

        # ── Latent Probes ────────────────────────────────────────────────
        self.latent_probes = nn.Parameter(torch.randn(num_latents, h_dim))
        if latent_init == "random":
            nn.init.xavier_uniform_(self.latent_probes)
        # Domain initialization would be done externally after model creation

        # ── Cross-Attention + Self-Attention Layers ──────────────────────
        ff_dim = h_dim * ff_multiplier
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(h_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.self_attn_layers = nn.ModuleList([
            SelfAttentionBlock(h_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # ── Refinement Classification Head ───────────────────────────────
        self.refine_head = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim // 2, 1),
        )

        # ── (Optional) Rescue Classification Head ────────────────────────
        # Same architecture, separate weights. Disabled by default for v1.
        self.rescue_head = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim // 2, 1),
        )

    def encode_reads(self, reads, context, mask):
        """
        Encode reads and context into token matrix.

        Args:
            reads: (B, n, d_read) — per-read features (padded)
            context: (B, d_context) — locus context
            mask: (B, n) — True for real reads

        Returns:
            tokens: (B, n+1, h) — read embeddings + context token
            token_mask: (B, n+1) — extended mask including context
        """
        B, n, _ = reads.shape

        # Encode each read
        read_embeddings = self.read_encoder(reads)  # (B, n, h)

        # Encode context as a single token
        ctx_token = self.context_encoder(context)    # (B, h)
        ctx_token = ctx_token.unsqueeze(1)           # (B, 1, h)

        # Concatenate: [read_1, read_2, ..., read_n, context]
        tokens = torch.cat([read_embeddings, ctx_token], dim=1)  # (B, n+1, h)

        # Extend mask: context token is always real
        ctx_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        token_mask = torch.cat([mask, ctx_mask], dim=1)  # (B, n+1)

        return tokens, token_mask

    def forward(self, reads, context, mask, mode="refine"):
        """
        Full forward pass.

        Args:
            reads: (B, n, d_read) — padded per-read features
            context: (B, d_context) — locus context
            mask: (B, n) — True for real reads, False for padding
            mode: "refine" or "rescue" — selects classification head

        Returns:
            logits: (B, 1) — raw logits (apply sigmoid externally)
            attn_weights: list of (B, num_heads, K, n+1) per layer
        """
        B = reads.shape[0]

        # ── Encode reads + context ───────────────────────────────────────
        tokens, token_mask = self.encode_reads(reads, context, mask)

        # ── Initialize latent probes (broadcast across batch) ────────────
        latents = self.latent_probes.unsqueeze(0).expand(B, -1, -1)
        # latents: (B, K, h)

        # ── Cross-attention + self-attention rounds ──────────────────────
        all_attn_weights = []
        for i in range(self.num_layers):
            latents, cross_attn_w = self.cross_attn_layers[i](
                latents, tokens, mask=token_mask
            )
            all_attn_weights.append(cross_attn_w)
            latents = self.self_attn_layers[i](latents)

        # ── Mean-pool latents → locus representation ─────────────────────
        z = latents.mean(dim=1)  # (B, h)

        # ── Classification head ──────────────────────────────────────────
        if mode == "refine":
            logits = self.refine_head(z)
        elif mode == "rescue":
            logits = self.rescue_head(z)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return logits, all_attn_weights

    def predict_proba(self, reads, context, mask, mode="refine"):
        """Convenience method: returns calibrated probabilities."""
        logits, _ = self.forward(reads, context, mask, mode=mode)
        return torch.sigmoid(logits)

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_weights(self, reads, context, mask):
        """
        Extract cross-attention weights for interpretability analysis.
        Returns list of (B, num_heads, K, n+1) tensors, one per layer.
        """
        with torch.no_grad():
            _, attn_weights = self.forward(reads, context, mask)
        return attn_weights


def build_model(config=None):
    """Build RIVA model from config dict."""
    if config is None:
        config = {}

    model = RIVAPerceiver(
        d_read=config.get("d_read", 18),
        d_context=config.get("d_context", 8),
        h_dim=config.get("hidden_dim", 64),
        num_latents=config.get("num_latents", 4),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_cross_attn_layers", 1),
        ff_multiplier=config.get("ffn_dim_multiplier", 2),
        dropout=config.get("dropout", 0.1),
        latent_init=config.get("latent_init", "random"),
    )

    print(f"[RIVA] Model built: {model.count_parameters():,} parameters",
          flush=True)

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model()
    print(model)

    # Simulate a batch: 4 loci with variable read counts
    B = 4
    max_n = 50
    d_read = 18
    d_context = 8

    reads = torch.randn(B, max_n, d_read)
    context = torch.randn(B, d_context)
    mask = torch.ones(B, max_n, dtype=torch.bool)
    # Simulate variable lengths
    mask[0, 30:] = False
    mask[1, 45:] = False
    mask[2, 20:] = False
    mask[3, 50:] = False  # Full

    logits, attn_weights = model(reads, context, mask)
    probs = torch.sigmoid(logits)

    print(f"\nInput shapes: reads={reads.shape}, context={context.shape}, mask={mask.shape}")
    print(f"Output: logits={logits.shape}, probs range=[{probs.min():.3f}, {probs.max():.3f}]")
    print(f"Attention weights: {len(attn_weights)} layers, shape={attn_weights[0].shape}")
    print(f"Total parameters: {model.count_parameters():,}")
