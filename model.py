import torch.nn as nn
import timm


class Model(nn.Module):
    def __init__(self, name, num_classes) -> None:
        super().__init__()
        self.batch_normal = nn.BatchNorm2d(1)
        # Black-white to 3-channel color
        self.color = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 10, 1, padding=0), 
            nn.ReLU(),
            nn.Conv2d(10, 3, 1, padding=0), 
            nn.ReLU()
        )
      
        self.core = timm.create_model(name, pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.batch_normal(x)
        x = self.color(x)
        x = self.core(x)
        return x
