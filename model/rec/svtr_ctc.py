import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from model.rec.svtr_encoder import SVTREncoder
from model.rec.tokenizer import Tokenizer
from model.rec.vocab import VOCAB


class SVTRCTC(nn.Module):
    """
    SVTR-CTC: Scene Text Recognition using Vision Transformer with CTC Loss
    
    Architecture:
    - SVTR Encoder: CNN + Transformer for feature extraction
    - Linear Classifier: Maps features to character probabilities
    - CTC Decoding: Alignment-free inference
    """
    def __init__(self, img_size=(32, 128), in_channels=3, 
                 embed_dim=[192, 256, 512],      # Base Configuration
                 enc_depth=[3, 9, 9],            # Blocks per stage
                 num_heads=[6, 8, 16],           # Heads per stage
                 charset=VOCAB, out_channels=512):
        super().__init__()
        
        self.tokenizer = Tokenizer(charset) 

        
        # Encoder
        self.encoder = SVTREncoder(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=enc_depth,
            num_heads=num_heads,
            out_channels=out_channels
        )
        
        # Classification head for CTC
        # Output: (B, T, num_classes) where T is sequence length
        self.head = nn.Linear(out_channels, self.tokenizer.num_classes)
        
        # For CTC, we need blank token at index 0
        self.blank_id = self.tokenizer.blank_id

    def forward(self, images, targets=None):
        """
        Args:
            images: (B, C, H, W) - Input images
            targets: (B, L) - Target token IDs (optional, for training)
        
        Returns:
            If training: log_probs (T, B, num_classes) for CTC loss
            If inference: log_probs for decoding
        """
        # Encoder: (B, C, H, W) -> (B, T, C)
        features = self.encoder(images)
        
        # Classification: (B, T, C) -> (B, T, num_classes)
        logits = self.head(features)
        
        # CTC expects log probabilities in shape (T, B, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)  # (B, T, V) -> (T, B, V)
        
        return log_probs

    @torch.inference_mode()
    def decode_greedy(self, images):
        """
        Greedy CTC decoding
        
        Args:
            images: (B, C, H, W)
        
        Returns:
            List of decoded strings
        """
        self.eval()
        log_probs = self.forward(images)  # (T, B, num_classes)
        return self.decode_probs(log_probs)

    def decode_probs(self, log_probs):
        """
        Decode from log probabilities.
        
        Args:
            log_probs: (T, B, num_classes)
            
        Returns:
            List of decoded strings
        """
        # Greedy decoding: take argmax at each timestep
        preds = log_probs.argmax(dim=-1)  # (T, B)
        preds = preds.permute(1, 0)  # (B, T)
        
        # CTC collapse: remove blanks and consecutive duplicates
        decoded = []
        for pred in preds:
            # Remove blanks and consecutive duplicates
            chars = []
            prev = None
            for p in pred.tolist():
                if p != self.blank_id and p != prev:
                    chars.append(p)
                prev = p
            decoded.append(chars)
        
        # Convert to strings
        texts = self.tokenizer.decode(decoded)
        return texts

