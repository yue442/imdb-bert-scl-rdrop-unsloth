"""SCL and R-Drop losses. No dependency on teacher files or external downloads."""
import torch
from torch import nn
from torch.nn import functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss, all-view anchors; input [B, V, H].

    Single-view classification uses V=1. Self pairs are removed. Anchors
    without any positive are excluded, not divided by zero. With no valid
    anchor, return a differentiable zero. This implementation uses tau as
    both temperature and base_temperature (no extra temperature scaling).
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, features, labels):
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, views, hidden]")
        b, views, hidden = features.shape
        labels = labels.reshape(-1)
        if len(labels) != b or min(b, views, hidden) < 1:
            raise ValueError("invalid labels or empty features")
        if not torch.isfinite(features).all():
            raise FloatingPointError("non-finite SCL features")
        # FP32 similarity and logsumexp even under mixed precision.
        z = F.normalize(features.float(), dim=-1).reshape(b * views, hidden)
        y = labels.repeat_interleave(views)
        not_self = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        positives = y[:, None].eq(y[None, :]) & not_self
        counts = positives.sum(1)
        valid = counts > 0
        if not valid.any():
            return features.float().sum() * 0.0
        # Casting inputs to float alone does NOT prevent autocast from
        # downcasting matmul; explicitly disable it for the similarity matrix.
        with torch.autocast(device_type=features.device.type, enabled=False):
            logits = z.float() @ z.float().T / self.temperature
        denominator = torch.logsumexp(logits.masked_fill(~not_self, -torch.inf), dim=1)
        log_prob = logits - denominator[:, None]
        # Select positives with masked_fill, avoiding 0 * inf for singleton batches.
        per_anchor = -log_prob.masked_fill(~positives, 0).sum(1) / counts.clamp_min(1)
        return per_anchor[valid].mean()


def symmetric_kl(logits_a, logits_b):
    """Mean of KL(p||q) and KL(q||p), each averaged over samples."""
    a, b = logits_a.float(), logits_b.float()
    return 0.5 * (
        F.kl_div(F.log_softmax(a, -1), F.softmax(b, -1), reduction="batchmean")
        + F.kl_div(F.log_softmax(b, -1), F.softmax(a, -1), reduction="batchmean")
    )
