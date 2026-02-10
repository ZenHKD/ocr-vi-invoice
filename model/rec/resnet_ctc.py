import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from model.rec.tokenizer import Tokenizer
from model.rec.vocab import VOCAB

class ResNetCTC(nn.Module):
    def __init__(self, name='resnet50', pretrained=True, in_channels=3, 
                 charset=VOCAB, out_channels=512):
        super(ResNetCTC, self).__init__()
        
        self.tokenizer = Tokenizer(charset)
        
        # Backbone
        if name == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            original_out_channels = 512
        elif name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            original_out_channels = 2048
        else:
            raise NotImplementedError(f"Backbone {name} not implemented")

        # Handle in_channels
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        def modify_stride(layer):
            """
            Modify the stride of the first block in the layer to preserve width.
            Changes stride (2, 2) -> (2, 1) for all Conv2d interactions (main path & downsample).
            """
            for m in layer[0].modules():
                if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                    m.stride = (2, 1)

        modify_stride(self.backbone.layer2)
        modify_stride(self.backbone.layer3)
        modify_stride(self.backbone.layer4)

        # Projection (Neck) to out_channels
        self.neck = nn.Conv2d(original_out_channels, out_channels, kernel_size=1)
        
        # CTC Head
        self.head = nn.Linear(out_channels, self.tokenizer.num_classes)
        self.blank_id = self.tokenizer.blank_id

    def forward(self, x, targets=None):
        # x: (B, C, H, W) -> (B, 3, 32, 256)
        
        # Extract features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x) # (B, 64, 16, 128) if W=256

        x = self.backbone.layer1(x)  # (B, C1, 16, 128)
        x = self.backbone.layer2(x)  # (B, C2, 8, 128) (stride 2,1)
        x = self.backbone.layer3(x)  # (B, C3, 4, 128) (stride 2,1)
        x = self.backbone.layer4(x)  # (B, C4, 2, 128) (stride 2,1)

        # Final pooling - collapse H to 1
        x = torch.mean(x, dim=2, keepdim=True) # (B, C4, 1, 128)
        
        # Adjusting if H was not reduced to 1 by strides alone (due to shapes)
        
        x = self.neck(x) # (B, out_channels, 1, W')
        
        # Prepare for CTC: (T, B, C)
        x = x.squeeze(2) # (B, C, W')
        x = x.permute(2, 0, 1) # (W', B, C) -> (T, B, C)
        
        # Head
        logits = self.head(x) # (T, B, num_classes)
        
        # Log Softmax
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1) # (T, B, num_classes)
        
        return log_probs

    @torch.inference_mode()
    def decode_greedy(self, images):
        self.eval()
        log_probs = self.forward(images)
        return self.decode_probs(log_probs)

    def decode_probs(self, log_probs):
        preds = log_probs.argmax(dim=-1) # (T, B)
        preds = preds.permute(1, 0) # (B, T)
        
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
