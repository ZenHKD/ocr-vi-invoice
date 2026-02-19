"""
SVTRv2 Combined Loss: CTC + SGM

Loss = L_ctc + λ₁ * L_sgm_left + λ₂ * L_sgm_right

where L_sgm uses cross-entropy on the SGM predictions.
SGM predictions are only available during training; when None, only CTC loss is used.
"""

import torch
import torch.nn as nn


class SVTRv2Loss(nn.Module):
    """
    Combined CTC + Semantic Guidance Module loss for SVTRv2.

    Args:
        blank: CTC blank token ID
        pad_id: Padding token ID
        lambda_sgm: Weight for SGM loss terms (default: 0.1 each)
    """
    def __init__(self, blank=0, pad_id=1, lambda_sgm=0.1, reduction='mean', zero_infinity=True):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id, reduction=reduction)
        self.blank = blank
        self.pad_id = pad_id
        self.lambda_sgm = lambda_sgm

    def forward(self, log_probs, targets, sgm_output=None,
                input_lengths=None, target_lengths=None):
        """
        Args:
            log_probs: (T, B, num_classes) — CTC log probabilities
            targets: (B, L) — target token IDs (padded)
            sgm_output: dict with 'sgm_left', 'sgm_right', 'sgm_targets' or None
            input_lengths: (B,) — optional, defaults to T for all
            target_lengths: (B,) — optional, computed from targets

        Returns:
            Total loss scalar
        """
        T, B, _ = log_probs.shape

        # ── CTC Loss ──
        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)

        if target_lengths is None:
            target_lengths = (targets != self.pad_id).sum(dim=1)
        else:
            if target_lengths.dim() > 1:
                target_lengths = target_lengths.squeeze()

        # Flatten targets for CTC
        targets_flat = []
        for i in range(B):
            length = target_lengths[i].item()
            targets_flat.extend(targets[i, :length].tolist())
        targets_flat = torch.tensor(targets_flat, dtype=torch.long, device=log_probs.device)

        loss_ctc = self.ctc_loss(log_probs, targets_flat, input_lengths, target_lengths)

        # ── SGM Loss (training only) ──
        if sgm_output is not None:
            sgm_left = sgm_output['sgm_left']     # (B, L, num_classes)
            sgm_right = sgm_output['sgm_right']   # (B, L, num_classes)
            sgm_targets = sgm_output['sgm_targets']  # (B, L)

            # Reshape for cross-entropy: (B*L, C) vs (B*L,)
            B_s, L_s, C = sgm_left.shape
            loss_sgm_left = self.ce_loss(
                sgm_left.reshape(B_s * L_s, C),
                sgm_targets.reshape(B_s * L_s)
            )
            loss_sgm_right = self.ce_loss(
                sgm_right.reshape(B_s * L_s, C),
                sgm_targets.reshape(B_s * L_s)
            )

            total_loss = loss_ctc + self.lambda_sgm * (loss_sgm_left + loss_sgm_right)
        else:
            total_loss = loss_ctc

        return total_loss
