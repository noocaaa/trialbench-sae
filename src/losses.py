"""
src/losses.py — Custom loss functions for clinical prediction.

AsymmetricBCELoss:
    Applies different penalties to false negatives (missed SAE) vs
    false positives (false alarm), reflecting the clinical reality
    that missing an adverse event is more costly than a false alarm.
    
    Uses soft weights (probability-based) rather than hard threshold
    for smooth gradients during training.
"""

import torch
import torch.nn as nn


class AsymmetricBCELoss(nn.Module):
    """
    Binary Cross-Entropy with asymmetric penalties for FN vs FP.

    Standard BCE treats FN and FP equally. In clinical settings,
    missing an SAE (FN) is typically more harmful than a false alarm (FP).

    Uses soft probability weighting:
    - FN weight: target * (1 - prob)  → high when target=1 and prob is low
    - FP weight: (1 - target) * prob  → high when target=0 and prob is high
    
    This creates a smooth loss landscape without hard threshold discontinuities.

    Parameters
    ----------
    fn_penalty : float — multiplier for false negative loss (default: 2.5)
    fp_penalty : float — multiplier for false positive loss (default: 1.0)
    pos_weight : float — class imbalance weight (like BCEWithLogitsLoss)
    """

    def __init__(self, fn_penalty=2.5, fp_penalty=1.0, pos_weight=None):
        super().__init__()
        self.fn_penalty = fn_penalty
        self.fp_penalty = fp_penalty
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        logits  : [batch] — raw model outputs (before sigmoid)
        targets : [batch] — binary labels (0 or 1)
        """
        # Standard BCE loss per sample
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )

        # Soft asymmetric weights based on predicted probability
        probs = torch.sigmoid(logits)
        
        # FN weight: high when target=1 and prob is low (missed positive)
        fn_weight = targets * (1.0 - probs)
        # FP weight: high when target=0 and prob is high (false alarm)
        fp_weight = (1.0 - targets) * probs
        
        # Combine: base weight 1.0 + additional penalty for errors
        # When fn_weight is high → apply fn_penalty
        # When fp_weight is high → apply fp_penalty
        asymmetric_factor = 1.0 + (self.fn_penalty - 1.0) * fn_weight + (self.fp_penalty - 1.0) * fp_weight
        
        return (bce * asymmetric_factor).mean()
