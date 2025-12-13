# models/losses.py
"""
Custom losses:
 - soft_jaccard(attn, target_mask): differentiable soft-jaccard between attention distribution and sequence-level
   target mask (both normalized to sum->1)
 - attention_sparsity_loss(attn): encourage sparse attention (L1 on attention or entropy)
"""

import torch
import torch.nn.functional as F

def soft_jaccard(attn: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """
    attn: (N,1) attention weights (non-negative), or raw scores (we'll softmax before)
    target: (N,) or (N,1) ground-truth soft target (0..1 per instance) - must be same length.
    Returns: scalar loss = 1 - soft_jaccard_index (so minimizing improves jaccard)
    """
    if attn.dim() == 2 and attn.shape[1] == 1:
        attn = attn.squeeze(1)
    attn = torch.clamp(attn, min=0.0)
    attn = attn / (attn.sum() + eps)

    target = target.view(-1).to(attn.device).float()
    target = torch.clamp(target, min=0.0)
    target = target / (target.sum() + eps)

    intersection = (attn * target).sum()
    union = (attn + target - attn * target).sum()
    jacc = (intersection + eps) / (union + eps)
    return 1.0 - jacc

def attention_sparsity_loss(attn: torch.Tensor, lam: float = 1e-3):
    """
    Encourage sparsity on attention weights; two options:
      - L1 on attention (sum(attn)) which is constant if attn normalized; so we use negative entropy to encourage peaky
      - negative entropy: -sum(p*log p) => minimized when p is deterministic; add lam factor
    attn: (N,1) softmaxed attention
    """
    if attn.dim() == 2 and attn.shape[1] == 1:
        attn = attn.squeeze(1)
    p = attn + 1e-12
    entropy = - (p * torch.log(p)).sum()
    return lam * entropy

# End of file
