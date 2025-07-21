# clip_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricCLIPLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, outputs, inputs=None):
        """
        Args:
            outputs: from CLIPModel, should contain:
                - logits_per_image (B x B)
                - logits_per_text (B x B)
            inputs: unused but kept for API consistency
        Returns:
            scalar loss (mean over batch)
        """
        logits_per_image = outputs.logits_per_image / self.temperature
        logits_per_text = outputs.logits_per_text / self.temperature

        batch_size = logits_per_image.size(0)
        labels = torch.arange(batch_size, device=logits_per_image.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        return (loss_i2t + loss_t2i) / 2
