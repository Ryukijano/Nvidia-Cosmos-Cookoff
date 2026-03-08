import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# User's ResNet variant (adapted for 2048-d features, no head)
class ResNet(nn.Module):
    def __init__(self, out_channels=4, has_fc=False):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=False)
        if not has_fc:
            self.resnet.fc = nn.Identity()  # Output 2048-d features
        else:
            # Keep the original fc layer for compatibility
            pass

    def forward(self, x):
        return self.resnet(x)
