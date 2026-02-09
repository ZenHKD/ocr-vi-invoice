import torch.nn as nn
from model.det.backbone import ResNet
from model.det.neck import FPN_ASF
from model.det.head import DBHead

class DBNetPP(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, in_channels=3, inner_channels=256, k=50, dcn=False):
        super(DBNetPP, self).__init__()
        self.backbone = ResNet(name=backbone, pretrained=pretrained, in_channels=in_channels, dcn=dcn)
        self.neck = FPN_ASF(self.backbone.out_channels, inner_channels=inner_channels)
        self.head = DBHead(inner_channels, k=k)

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        predictions = self.head(features)
        return predictions
