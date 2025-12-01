import torch
import torch.nn.functional as F


def clip_style_alignment_loss(rast_z: torch.Tensor, vect_z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """CLIP-style symmetric cross entropy loss between two modalities.

    Args:
        rast_z: Tensor of shape [B, D], assumed to be L2 normalized.
        vect_z: Tensor of shape [B, D], assumed to be L2 normalized.
        temperature: Scaling factor for logits.
    """
    logits = torch.matmul(rast_z, vect_z.t()) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_r2v = F.cross_entropy(logits, targets)
    loss_v2r = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_r2v + loss_v2r)
