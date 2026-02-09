import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from model.det.dcn import DeformableConv2d

class ResNet(nn.Module):
    def __init__(self, name='resnet50', pretrained=True, in_channels=3, dcn=False):
        super(ResNet, self).__init__()

        if name == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            self.out_channels = [64, 128, 256, 512]
        elif name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError(f"Backbone {name} not implemented")

        # Handle different input channels if not 3
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Apply DCN to stages c3, c4, c5 (layer2, layer3, layer4)
        if dcn:
            self._apply_dcn(self.model.layer2)
            self._apply_dcn(self.model.layer3)
            self._apply_dcn(self.model.layer4)

        # Layers to extract features from
        self.layer1 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool, self.model.layer1)
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

    def _apply_dcn(self, layer):
        for i, block in enumerate(layer):
            if hasattr(block, 'conv2'):
                # Replace 3x3 conv (conv2) with DCN
                # Conv2 is typically the middle layer in Bottleneck (1x1 -> 3x3 -> 1x1)
                in_channels = block.conv2.in_channels
                out_channels = block.conv2.out_channels
                stride = block.conv2.stride
                padding = block.conv2.padding
                
                # Careful: torchvision ResNet Bottleneck stride might be tuple or int
                if isinstance(stride, tuple): stride = stride[0]
                if isinstance(padding, tuple): padding = padding[0]

                block.conv2 = DeformableConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]
