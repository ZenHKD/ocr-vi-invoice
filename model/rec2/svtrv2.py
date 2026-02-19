"""
SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition

Paper: https://arxiv.org/abs/2411.15858
Reference: https://github.com/Topdu/OpenOCR

Architecture:
    ConvStem → 3-stage backbone (Local + Global mixing) → FRM → CTC Head
    + SGM (training only, discarded at inference)

Variants:
    - Tiny:  D=[64,128,256],  Blocks=[3,6,3],  Mixer=[L3, L3G3, G3]
    - Small: D=[96,192,256],  Blocks=[3,6,6],  Mixer=[L3, L3G3, G6]
    - Base:  D=[128,256,384], Blocks=[3,6,6],  Mixer=[L3, L2G4, G6]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rec2.tokenizer import Tokenizer
from model.rec2.vocab import VOCAB


# ──────────────────────────── Building Blocks ────────────────────────────

class MLP(nn.Module):
    """Feed-forward with expansion ratio."""
    def __init__(self, dim, expansion=4, dropout=0.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class LocalMixing(nn.Module):
    """
    Two consecutive grouped depthwise-separable convolutions (3x3).
    Captures local character features (edges, textures, strokes).
    """
    def __init__(self, dim):
        super().__init__()
        groups = max(dim // 32, 1)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(dim)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=groups)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act2 = nn.GELU()

    def forward(self, x, H, W):
        """x: (B, N, D) where N = H*W"""
        B, N, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, H, W)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


class GlobalMixing(nn.Module):
    """Multi-Head Self-Attention for global context."""
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.num_heads = max(dim // 32, 1)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, H, W):
        """x: (B, N, D)"""
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class MixingBlock(nn.Module):
    """Pre-LN → Mixing (Local or Global) → Residual → Pre-LN → MLP → Residual"""
    def __init__(self, dim, is_local=True, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = LocalMixing(dim) if is_local else GlobalMixing(dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, expansion=4, dropout=dropout)

    def forward(self, x, H, W):
        x = x + self.mixer(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


# ──────────────────────────── Embeddings ─────────────────────────────────

class ConvStem(nn.Module):
    """Two 3x3 convolutions with stride 2 → downsample 4x in both H and W."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(mid, out_channels, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x):
        """x: (B, C, H, W) → (B, D0, H/4, W/4)"""
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class PatchMerging(nn.Module):
    """Downsample height by 2, keep width. Maps dim_in → dim_out."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=(2, 1), padding=1)
        self.norm = nn.BatchNorm2d(dim_out)

    def forward(self, x, H, W):
        """x: (B, H*W, D_in) → (B, H/2*W, D_out)"""
        B, _, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, H, W)
        x = self.norm(self.conv(x))  # (B, D_out, H/2, W)
        new_H, new_W = x.shape[2], x.shape[3]
        return x.flatten(2).transpose(1, 2), new_H, new_W


# ──────────────────────────── SVTRv2 Backbone ────────────────────────────

class SVTRStage(nn.Module):
    """One stage of the SVTRv2 backbone with N mixing blocks."""
    def __init__(self, dim, num_blocks, num_local, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            is_local = (i < num_local)
            self.blocks.append(MixingBlock(dim, is_local=is_local, dropout=dropout))

    def forward(self, x, H, W):
        for block in self.blocks:
            x = block(x, H, W)
        return x


# ──────────────────────────── Feature Rearrangement Module ───────────────

class FRM(nn.Module):
    """
    Feature Rearrangement Module.
    1. Horizontal self-attention per row to rearrange within each horizontal strip
    2. Vertical cross-attention with learnable selecting token to collapse height
    Output: (B, W', D)
    """
    def __init__(self, dim, num_heads=None, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads or max(dim // 32, 1)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Horizontal self-attention
        self.h_qkv = nn.Linear(dim, dim * 3)
        self.h_proj = nn.Linear(dim, dim)
        self.h_norm = nn.LayerNorm(dim)
        self.h_mlp = MLP(dim)
        self.h_norm2 = nn.LayerNorm(dim)

        # Vertical cross-attention with selecting token
        self.select_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.v_q = nn.Linear(dim, dim)
        self.v_kv = nn.Linear(dim, dim * 2)
        self.v_proj = nn.Linear(dim, dim)
        self.v_norm_q = nn.LayerNorm(dim)
        self.v_norm_kv = nn.LayerNorm(dim)
        self.v_mlp = MLP(dim)
        self.v_norm2 = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, H, W):
        """
        x: (B, H*W, D) visual features from backbone
        Returns: (B, W, D) character-aligned feature sequence
        """
        B, N, D = x.shape

        # ─── 1. Horizontal rearrangement (self-attention per row) ───
        x = x.reshape(B, H, W, D)
        # Process each row: (B*H, W, D)
        x_rows = x.reshape(B * H, W, D)
        x_rows_normed = self.h_norm(x_rows)

        qkv = self.h_qkv(x_rows_normed).reshape(B * H, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*H, heads, W, hd)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B * H, W, D)
        x_rows = x_rows + self.h_proj(out)
        x_rows = x_rows + self.h_mlp(self.h_norm2(x_rows))

        # Reshape back: (B, H, W, D)
        x_h = x_rows.reshape(B, H, W, D)

        # ─── 2. Vertical rearrangement (cross-attention per column) ───
        # Transpose to column-wise: for each column j, gather features from all rows
        # x_h: (B, H, W, D) → process column-by-column
        # Expand selecting token: (B, W, D)
        t_s = self.select_token.expand(B, W, -1)

        # For cross-attention: query = selecting token, key/value = column features
        # Reshape: (B*W, 1, D) for query, (B*W, H, D) for key/value
        x_cols = x_h.permute(0, 2, 1, 3).reshape(B * W, H, D)  # (B*W, H, D)
        t_q = t_s.reshape(B * W, 1, D)

        t_q_normed = self.v_norm_q(t_q)
        x_cols_normed = self.v_norm_kv(x_cols)

        q = self.v_q(t_q_normed).reshape(B * W, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.v_kv(x_cols_normed).reshape(B * W, H, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B * W, 1, D)
        t_q = t_q + self.v_proj(out)
        t_q = t_q + self.v_mlp(self.v_norm2(t_q))

        # (B*W, 1, D) → (B, W, D)
        return t_q.reshape(B, W, D)


# ──────────────────────────── Semantic Guidance Module ────────────────────

class SGM(nn.Module):
    """
    Semantic Guidance Module (training only).
    Encodes left/right character context strings and cross-attends to visual features.
    Discarded during inference.
    """
    def __init__(self, dim, num_classes, context_window=3, num_heads=None, dropout=0.0):
        super().__init__()
        self.context_window = context_window
        self.num_heads = num_heads or max(dim // 32, 1)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Character embedding for context
        self.char_embed = nn.Embedding(num_classes, dim)

        # Context encoder (small transformer)
        self.context_norm = nn.LayerNorm(dim)
        self.context_attn = nn.MultiheadAttention(dim, self.num_heads, dropout=dropout, batch_first=True)
        self.context_mlp = MLP(dim, expansion=2, dropout=dropout)
        self.context_norm2 = nn.LayerNorm(dim)

        # Learnable tokens for left and right
        self.left_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.right_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Cross-attention: context query → visual features
        self.cross_q = nn.Linear(dim, dim)
        self.cross_kv = nn.Linear(dim, dim * 2)
        self.cross_proj = nn.Linear(dim, dim)
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)

        # Classifier
        self.sgm_head = nn.Linear(dim, num_classes)

        self.attn_drop = nn.Dropout(dropout)

    def _encode_context(self, context_ids, direction_token):
        """
        context_ids: (B, L, context_window) — character IDs for context
        direction_token: (1, 1, D) — left or right token
        Returns: (B, L, D) — encoded context representations
        """
        B, L, W = context_ids.shape
        D = direction_token.shape[-1]

        # Embed context characters: (B, L, W, D)
        ctx_embed = self.char_embed(context_ids)

        # Add direction token: (B, L, W, D)
        ctx_embed = ctx_embed + direction_token.unsqueeze(0)

        # Reshape for batch processing: (B*L, W, D)
        ctx_flat = ctx_embed.reshape(B * L, W, D)

        # Self-attention to encode context
        ctx_normed = self.context_norm(ctx_flat)
        ctx_out, _ = self.context_attn(ctx_normed, ctx_normed, ctx_normed)
        ctx_flat = ctx_flat + ctx_out
        ctx_flat = ctx_flat + self.context_mlp(self.context_norm2(ctx_flat))

        # Pool to single vector: mean pool over context window
        ctx_pooled = ctx_flat.mean(dim=1)  # (B*L, D)
        return ctx_pooled.reshape(B, L, D)

    def _cross_attend(self, queries, visual_features):
        """
        queries: (B, L, D) — context queries
        visual_features: (B, N, D) — flattened visual features
        Returns: (B, L, D)
        """
        B, L, D = queries.shape
        N = visual_features.shape[1]

        q = self.cross_q(self.cross_norm_q(queries))
        kv = self.cross_kv(self.cross_norm_kv(visual_features))

        q = q.reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = kv.reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.cross_proj(out)

    def forward(self, visual_features, targets, pad_id=1):
        """
        visual_features: (B, H*W, D) — visual features from backbone (before FRM)
        targets: (B, L) — target character IDs

        Returns:
            sgm_left: (B, L, num_classes) — logits from left context
            sgm_right: (B, L, num_classes) — logits from right context
            sgm_targets: (B, L) — target labels for SGM loss
        """
        B, L = targets.shape
        ws = self.context_window

        # Build left and right context for each character position
        # Pad targets with pad_id
        padded = F.pad(targets, (ws, ws), value=pad_id)  # (B, L + 2*ws)

        left_ctx = []
        right_ctx = []
        for i in range(L):
            # left context: characters before position i
            left_ctx.append(padded[:, i:i + ws])           # (B, ws)
            # right context: characters after position i
            right_ctx.append(padded[:, i + ws + 1:i + 2 * ws + 1])  # (B, ws)

        left_ctx = torch.stack(left_ctx, dim=1)   # (B, L, ws)
        right_ctx = torch.stack(right_ctx, dim=1)  # (B, L, ws)

        # Encode left and right contexts
        left_encoded = self._encode_context(left_ctx, self.left_token)    # (B, L, D)
        right_encoded = self._encode_context(right_ctx, self.right_token)  # (B, L, D)

        # Cross-attend to visual features
        left_feat = left_encoded + self._cross_attend(left_encoded, visual_features)
        right_feat = right_encoded + self._cross_attend(right_encoded, visual_features)

        # Classify
        sgm_left = self.sgm_head(left_feat)    # (B, L, num_classes)
        sgm_right = self.sgm_head(right_feat)  # (B, L, num_classes)

        return {
            'sgm_left': sgm_left,
            'sgm_right': sgm_right,
            'sgm_targets': targets,
        }


# ──────────────────────────── SVTRv2 Model ───────────────────────────────

# Variant configurations: (dims, num_blocks, num_local_per_stage)
VARIANTS = {
    'tiny': {
        'dims': [64, 128, 256],
        'num_blocks': [3, 6, 3],
        'num_local': [3, 3, 0],    # L3, L3G3, G3
    },
    'small': {
        'dims': [96, 192, 256],
        'num_blocks': [3, 6, 6],
        'num_local': [3, 3, 0],    # L3, L3G3, G6
    },
    'base': {
        'dims': [128, 256, 384],
        'num_blocks': [3, 6, 6],
        'num_local': [3, 2, 0],    # L3, L2G4, G6
    },
}


class SVTRv2(nn.Module):
    """
    SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition

    Args:
        variant: 'tiny', 'small', or 'base'
        in_channels: Number of input image channels (default: 3)
        charset: Character set for the tokenizer
        dropout: Dropout rate
        context_window: SGM context window size
    """
    def __init__(self, variant='small', in_channels=3, charset=VOCAB,
                 dropout=0.0, context_window=3):
        super().__init__()

        assert variant in VARIANTS, f"Unknown variant: {variant}. Choose from {list(VARIANTS.keys())}"
        cfg = VARIANTS[variant]
        dims = cfg['dims']
        num_blocks = cfg['num_blocks']
        num_local = cfg['num_local']

        self.tokenizer = Tokenizer(charset)
        self.dims = dims

        # ── Patch Embedding ──
        self.stem = ConvStem(in_channels, dims[0])

        # ── 3-Stage Backbone ──
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()

        for i in range(3):
            self.stages.append(SVTRStage(dims[i], num_blocks[i], num_local[i], dropout))
            if i < 2:
                self.merges.append(PatchMerging(dims[i], dims[i + 1]))

        self.backbone_norm = nn.LayerNorm(dims[2])

        # ── Feature Rearrangement Module ──
        self.frm = FRM(dims[2], dropout=dropout)

        # ── Semantic Guidance Module (training only) ──
        self.sgm = SGM(dims[2], self.tokenizer.num_classes,
                        context_window=context_window, dropout=dropout)

        # ── CTC Head ──
        self.head = nn.Linear(dims[2], self.tokenizer.num_classes)
        self.blank_id = self.tokenizer.blank_id

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def extract_features(self, x):
        """
        Extract visual features from image.

        Args:
            x: (B, C, H, W)

        Returns:
            features: (B, H'*W', D) — flattened visual features
            H': height after backbone
            W': width after backbone
        """
        # Stem: (B, C, H, W) → (B, D0, H/4, W/4)
        x = self.stem(x)
        B, D, H, W = x.shape

        # Flatten to sequence: (B, H*W, D)
        x = x.flatten(2).transpose(1, 2)

        # 3 stages with patch merging
        for i, stage in enumerate(self.stages):
            x = stage(x, H, W)
            if i < 2:
                x, H, W = self.merges[i](x, H, W)

        x = self.backbone_norm(x)
        return x, H, W

    def forward(self, x, targets=None):
        """
        Args:
            x: (B, C, H, W) — input images, e.g. (B, 3, 32, 256)
            targets: (B, L) — target token IDs (only needed for SGM during training)

        Returns:
            If targets is None (inference):
                log_probs: (T, B, num_classes) — CTC log probabilities
            If targets is provided (training):
                (log_probs, sgm_output) where sgm_output is a dict with
                'sgm_left', 'sgm_right', 'sgm_targets'
        """
        # Extract backbone features
        features, H, W = self.extract_features(x)  # (B, H'*W', D)

        # SGM branch (training only)
        sgm_output = None
        if targets is not None and self.training:
            sgm_output = self.sgm(features, targets, pad_id=self.tokenizer.pad_id)

        # FRM: rearrange to character sequence
        char_features = self.frm(features, H, W)  # (B, W', D)

        # CTC Head
        logits = self.head(char_features)  # (B, W', num_classes)

        # Rearrange to CTC format: (T, B, num_classes)
        logits = logits.permute(1, 0, 2)  # (T, B, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)

        if sgm_output is not None:
            return log_probs, sgm_output
        return log_probs

    @torch.inference_mode()
    def decode_greedy(self, images):
        """Decode images to text strings using greedy CTC decoding."""
        self.eval()
        log_probs = self.forward(images)
        return self.decode_probs(log_probs)

    def decode_probs(self, log_probs):
        """
        Greedy CTC decoding on log_probs.

        Args:
            log_probs: (T, B, num_classes)

        Returns:
            List of decoded strings
        """
        preds = log_probs.argmax(dim=-1)  # (T, B)
        preds = preds.permute(1, 0)       # (B, T)

        decoded = []
        for pred in preds:
            chars = []
            prev = None
            for p in pred.tolist():
                if p != self.blank_id and p != prev:
                    chars.append(p)
                prev = p
            decoded.append(chars)

        texts = self.tokenizer.decode(decoded)
        return texts
