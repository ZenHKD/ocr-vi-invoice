import torch
import torch.nn as nn

class CTCLoss(nn.Module):
    """
    CTC Loss for SVTR-CTC model
    
    CTC (Connectionist Temporal Classification) allows training without 
    explicit alignment between input and output sequences.
    """
    def __init__(self, blank=0, pad_id=1, reduction='mean', zero_infinity=True):
        super().__init__()
        self.loss_fn = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        self.blank = blank
        self.pad_id = pad_id

    def forward(self, log_probs, targets, input_lengths=None, target_lengths=None):
        """
        Args:
            log_probs: (T, B, num_classes) - Log probabilities from model
            targets: (B, L) - Target sequences (padded)
            input_lengths: (B,) - Length of each sequence in log_probs
            target_lengths: (B,) - Length of each target sequence
        
        Returns:
            CTC loss value
        """
        T, B, _ = log_probs.shape
        
        # If lengths not provided, assume full length
        if input_lengths is None:
            input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)
        
        if target_lengths is None:
            # Calculate actual target lengths (excluding pad tokens)
            target_lengths = (targets != self.pad_id).sum(dim=1)
        else:
            # Ensure target_lengths is 1D tensor
            if target_lengths.dim() > 1:
                target_lengths = target_lengths.squeeze()
        
        # Flatten targets for CTC 
        targets_flat = []
        for i in range(B):
            length = target_lengths[i].item()
            targets_flat.extend(targets[i, :length].tolist())
        targets_flat = torch.tensor(targets_flat, dtype=torch.long, device=log_probs.device)
    
        return self.loss_fn(log_probs, targets_flat, input_lengths, target_lengths)

