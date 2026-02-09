import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import Block, Mlp, Attention


class ConvBNLayer(nn.Module):
    """Convolution + Batch Normalization + Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act() if act is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



class LocalMixing(nn.Module):
    """
    Local Mixing Module for SVTRv2.
    Performs local feature mixing using grouped convolutions.
    """
    def __init__(self, dim, num_heads=0, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True),
            act_layer(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        return self.mixer(x) 
        


class SVTRBlock(nn.Module):
    """
    Unified block that can be either Local Mixing (Conv) or Global Mixing (Attn).
    """
    def __init__(self, dim, num_heads, mixing_type='local', mlp_ratio=4., qkv_bias=True, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.mixing_type = mixing_type
        self.norm1 = norm_layer(dim)
        
        if mixing_type == 'local':
            # Local Mixing
            self.mixer = LocalMixing(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer)
        else:
            # Global Mixing: Multi-Head Self Attention (timm implementation)
            self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        """
        x: (B, N, C)
        """
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)
        
        if self.mixing_type == 'local':
            # Reshape to (B, C, H, W)
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.mixer(x)
            # Reshape back to (B, N, C)
            x = x.flatten(2).transpose(1, 2)
        else:
            x = self.mixer(x)
            
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer.
    Strided convolution to downsample.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(2, 1), padding=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, C', H/2, W)
        """
        x = self.conv(x)
        # Permute to (B, H, W, C) for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Permute back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class SVTRStage(nn.Module):
    def __init__(self, dim, depth, num_heads, mixing_types, mlp_ratio=4., qkv_bias=True, drop_path=0.):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            mixing_type = mixing_types[i]
            self.blocks.append(
                SVTRBlock(
                    dim=dim, 
                    num_heads=num_heads, 
                    mixing_type=mixing_type, 
                    mlp_ratio=mlp_ratio, 
                    qkv_bias=qkv_bias,
                    drop_path=drop_path
                )
            )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, N, C)
        
        for block in self.blocks:
            x = block(x, H, W)
            
        # Reshape back to feature map
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class SVTREncoder(nn.Module):
    """
    SVTRv2 Encoder.
    """
    def __init__(
        self, 
        img_size=(32, None), 
        in_channels=3, 
        embed_dim=[192, 256, 512], 
        depth=[3, 9, 9],
        num_heads=[6, 8, 16],
        mixer_types=['Local']*3 + ['Local']*4 + ['Global']*5 + ['Global']*9,
        mlp_ratio=4., 
        qkv_bias=True, 
        out_channels=None
    ):
        super().__init__()
        
        # Normalize config inputs
        if not isinstance(embed_dim, (list, tuple)): embed_dim = [embed_dim] * 3
        if not isinstance(depth, (list, tuple)): depth = [depth] * 3
        if not isinstance(num_heads, (list, tuple)): num_heads = [num_heads] * 3
        
        self.out_channels = out_channels if out_channels else embed_dim[-1]
        self.num_stages = len(embed_dim)

        self.stem1 = nn.Sequential(
            ConvBNLayer(in_channels, 32, kernel_size=3, stride=2, padding=1),
            ConvBNLayer(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.stem2 = nn.Sequential(
            ConvBNLayer(64, 64, kernel_size=3, stride=2, padding=1),
            ConvBNLayer(64, 128, kernel_size=3, stride=1, padding=1),
        )
        self.stem3 = nn.Sequential(
            ConvBNLayer(128, 128, kernel_size=3, stride=(2, 1), padding=1), 
            ConvBNLayer(128, embed_dim[0], kernel_size=3, stride=1, padding=1),
        )

        # Positional Embedding
        # Stem reduces H by 8, W by 4
        # Assuming max_H=32 => feat_H=4
        # Assuming max_W=384 (or larger) => feat_W=96
        # We use a large enough width to cover most cases, e.g. 512px / 4 = 128
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim[0], 4, 128)) 
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        
        stage_mixers = [
            ['local'] * depth[0],
            ['local'] * (depth[1]//2) + ['global'] * (depth[1] - depth[1]//2),
            ['global'] * depth[2]
        ]

        for i in range(self.num_stages):
            stage = SVTRStage(
                dim=embed_dim[i],
                depth=depth[i],
                num_heads=num_heads[i],
                mixing_types=stage_mixers[i],

                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias
            )
            self.stages.append(stage)
            
            if i < self.num_stages - 1:
                # Patch Merge: H/2, W/1 (preserve width)
                merge = PatchMerging(embed_dim[i], embed_dim[i+1])
                self.merges.append(merge)
        
        # Late Pooling
        # Collapses H dimension to 1.
        self.pool = nn.AdaptiveAvgPool2d((1, None)) 
        
        self.norm = nn.LayerNorm(embed_dim[-1], eps=1e-6)
        self.proj = nn.Linear(embed_dim[-1], self.out_channels) if embed_dim[-1] != self.out_channels else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x: (B, 3, H, W)
        
        # Stem
        x = self.stem1(x)
        x = self.stem2(x)
        x = self.stem3(x) 
        # Feature map is now (B, D0, H/8, W/4)
        
        # Add Positional Embedding
        # Resize/Slice pos_embed to match x
        if self.pos_embed.shape[2:] != x.shape[2:]:
            # If size mismatch, interpolate or slice
            # Here we assume H is fixed at 4 (input 32). W is variable.
            # We slice W if x is smaller, or interpolate if x is larger
            H_feat, W_feat = x.shape[2], x.shape[3]
            pos_embed = self.pos_embed
            
            if H_feat != pos_embed.shape[2] or W_feat > pos_embed.shape[3]:
                 # Interpolate if H differs or W is too large
                 pos_embed = F.interpolate(pos_embed, size=(H_feat, W_feat), mode='bilinear', align_corners=True)
            else:
                 # Slice if W is within bounds
                 pos_embed = pos_embed[:, :, :, :W_feat]
            
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        
        for i in range(self.num_stages):
            x = self.stages[i](x)
            if i < self.num_stages - 1:
                x = self.merges[i](x)
        
        # x: (B, D_last, H_final, W_final)
        # Collapse Height
        x = self.pool(x) # (B, D_last, 1, W_final)
        x = x.flatten(2).transpose(1, 2) # (B, W_final, D_last)
        
        x = self.norm(x)
        x = self.proj(x)
        
        return x

