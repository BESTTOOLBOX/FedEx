import torch
from torch import nn
 
 
# 创建fire module模块儿
class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        # torch.cat 将self.expand1x1_activation(self.expand1x1(x))与self.expand3x3_activation(self.expand3x3(x))通道数叠加
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )
 
 
# 针对squeezenet1_0版本，创建SqueezeNet框架。
class SqueezeNet_header(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super(SqueezeNet_header, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 3x32x32 -> 64x32x32
            nn.ReLU(inplace=True),
            Fire(64, 16, 64, 64),  # 64x32x32 -> 128x32x32
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 128x32x32 -> 128x16x16
            Fire(128, 32, 64, 64),  # 128x16x16 -> 128x16x16
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 128x16x16 -> 128x8x8
            Fire(128, 64, 128, 128),  # 128x8x8 -> 256x8x8
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 256x8x8 -> 256x4x4
            Fire(256, 64, 256, 256)  # 256x4x4 -> 512x4x4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(512, self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 512x4x4 -> 10x1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # torch.Size([1, 512, 4, 4])
        y = self.classifier(x)  # torch.Size([1, 10, 1, 1])
        y = y.view(x.size(0), -1)  # torch.Size([1, 10])
        return x.view(x.size(0), -1), y
 
 
