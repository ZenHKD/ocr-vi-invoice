import torch
import torch.nn as nn
import torchvision.ops as ops

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(DeformableConv2d, self).__init__()
        
        # Offset + Mask layer
        # Output channels: 2 * kernel_name * kernel_size (offsets) + kernel_size * kernel_size (masks)
        # For 3x3 kernel: 2*9=18 offsets + 9 masks = 27 channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.offset_mask_conv = nn.Conv2d(
            in_channels, 
            3 * kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation,
            bias=True
        )
        
        # Initialize weights for offset/mask conv
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        offset_mask = self.offset_mask_conv(x)
        
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        
        out = ops.deform_conv2d(
            x, 
            offset, 
            self.weight, 
            self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation,
            mask=mask
        )
        
        return out
