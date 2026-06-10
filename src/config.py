# src/config.py — Shared training configuration for all deep learning models
# All models use the same hyperparameters for fair comparison.

# ── Reproducibility ───────────────────────────────────────────────
RANDOM_STATE = 42

# ── Training ──────────────────────────────────────────────────────
EPOCHS     = 100
BATCH_SIZE = 32
LR         = 3e-4
WEIGHT_DECAY = 1e-4

# ── Optimizer ─────────────────────────────────────────────────────
OPTIMIZER  = "adamw"
SCHEDULER  = "cosine"
GRAD_CLIP  = 1.0

# ── Architecture defaults ─────────────────────────────────────────
DROPOUT    = 0.3

# ── Early Stopping ────────────────────────────────────────────────
VAL_SPLIT  = 0.125     # Fraction of training data used for validation (early stopping)
                       # In nested CV: 10% of total data = 12.5% of the 80% outer train
PATIENCE   = 5         # Epochs with no improvement before stopping
INNER_PATIENCE = 5     # Patience for inner loop (threshold calibration)
INNER_EPOCHS = 30      # Max epochs for inner loop

# ── Nested Cross-Validation ───────────────────────────────────────
OUTER_FOLDS = 5        # Number of outer folds (test sets)
INNER_FOLDS = 3        # Number of inner folds (threshold calibration)

# ── Threshold Calibration ─────────────────────────────────────────
TUNE_THRESHOLD = True           # If True, find optimal threshold on calibration set
THRESHOLD_CRITERION = "f1"      # Metric to optimize: "f1" or "youden"
CAL_SPLIT = 0.15       # Fraction of training data for threshold calibration

# ── Asymmetric Medical Loss ───────────────────────────────────────
# BCEWithLogitsLoss with pos_weight handles class imbalance.
# For true asymmetric loss (different penalties for FN vs FP),
# use a custom loss that applies FN_PENALTY to false negatives.
FN_PENALTY = 2.5       # Multiplier for False Negative loss (missed SAE)
FP_PENALTY = 1.0       # Multiplier for False Positive loss (false alarm)
USE_ASYMMETRIC_LOSS = False  # If True, use custom AsymmetricBCELoss; else standard BCEWithLogitsLoss
