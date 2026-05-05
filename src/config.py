# src/config.py — Shared training configuration for all deep learning models
# All models use the same hyperparameters for fair comparison.

# ── Training ──────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 64
LR         = 3e-4
WEIGHT_DECAY = 1e-4

# ── Optimizer ─────────────────────────────────────────────────────
OPTIMIZER  = "adamw"
SCHEDULER  = "cosine"
GRAD_CLIP  = 1.0

# ── Architecture defaults ─────────────────────────────────────────
DROPOUT    = 0.3