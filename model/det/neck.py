import torch
import torch.nn as nn
import torch.nn.functional as F
from model.det.layers import ConvBnRelu

class FPN_ASF(nn.Module):
    def __init__(self, in_channels_list, inner_channels=256, use_asf=True):
        super(FPN_ASF, self).__init__()
        self.use_asf = use_asf
        self.inner_channels = inner_channels
        self.in_channels_list = in_channels_list

        # Lateral connections (transform backbone features to inner_channels)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, inner_channels, 1))

        # Output connections (smooth aliasing effect)
        self.fpn_convs = nn.ModuleList()
        for _ in in_channels_list:
            self.fpn_convs.append(ConvBnRelu(inner_channels, inner_channels, 3, 1, 1))

        if self.use_asf:
            self.asf = ScaleFeatureSelection(len(in_channels_list), inner_channels)

    def forward(self, x):
        # x: [c2, c3, c4, c5]

        # Top-down pathway
        # Start from the last layer (c5)
        last_inner = self.lateral_convs[-1](x[-1])
        results = [self.fpn_convs[-1](last_inner)]

        # Go backwards from second to last
        for idx in range(len(x) - 2, -1, -1):
            inner_top_down = F.interpolate(last_inner, size=x[idx].shape[-2:], mode="nearest")
            lateral = self.lateral_convs[idx](x[idx])
            last_inner = lateral + inner_top_down
            results.insert(0, self.fpn_convs[idx](last_inner))

        # results: [p2, p3, p4, p5] all with inner_channels

        if self.use_asf:
             return self.asf(results)

        return torch.cat(results, dim=1) # Fallback if ASF is not used (not typical for DBNet++)


class ScaleFeatureSelection(nn.Module):
    def __init__(self, num_levels, inner_channels):
        super(ScaleFeatureSelection, self).__init__()
        self.inner_channels = inner_channels

        # Attention generation
        self.conv_atten = nn.Conv2d(inner_channels * num_levels, num_levels, 1)

    def forward(self, features):
        # features: [p2, p3, p4, p5]
        # Upsample all to the size of p2 (the largest one)
        target_size = features[0].shape[-2:]

        upsampled_features = []
        for i, feature in enumerate(features):
            if i > 0:
                feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=True)
            upsampled_features.append(feature)

        # Concatenate: (N, C*4, H, W)
        concat_features = torch.cat(upsampled_features, dim=1)

        # Attention map: (N, 4, H, W)
        score = F.softmax(self.conv_atten(concat_features), dim=1)

        # Weighted sum
        out = 0
        for i in range(len(upsampled_features)):
            out += upsampled_features[i] * score[:, i:i+1, :, :]

        return out
