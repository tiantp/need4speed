# A modified ResNet used for speeding up CIFAR10 training.
# Reference :
#      https://github.com/davidcpage/cifar10-fast
#      https://myrtle.ai/learn/how-to-train-your-resnet-1-baseline/
#      https://lambdalabs.com/blog/resnet9-train-to-94-cifar10-accuracy-in-100-seconds/

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

"""
The modified Blocks from David Page
  https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
"""
class Block(nn.Module):
    def __init__(self, inplanes: int, outplanes: int, stride: int=1) -> None:
        ''' add_sideconv : For blocks where number of outplane channel is larger
        than inplane channel, a side convolution of 1x1 is applied to ensure
        the channels are matching before adding in the skip layer '''
        super(Block, self).__init__()
        self.add_sideconv = (inplanes != outplanes) or (stride != 1)
        self.bn1    = nn.BatchNorm2d(inplanes)
        self.relu1  = nn.ReLU()
        self.c3x3_A = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride,
                            padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(outplanes)
        self.relu2  = nn.ReLU()
        self.c3x3_B = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1,
                            padding=1, bias=False)
        self.c1x1   = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride,
                            padding=0, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        y  = self.relu1(self.bn1(x))
        z  = self.relu2(self.bn2(self.c3x3_A(y)))
        z  = self.c3x3_B(z)
        z += self.c1x1(y) if self.add_sideconv else y
        return z

class Final(nn.Module):
    def __init__(self, inplanes: int, out_features: int):
        super(Final, self).__init__()
        self.avgpool = nn.AvgPool2d(4) # [N, inplanes, H/4, W/4]
        self.maxpool = nn.MaxPool2d(4) # [N, inplanes, H/4, W/4]
        self.flatten = nn.Flatten()
        self.linear  = nn.Linear(2 * inplanes, out_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.maxpool(x)
        y2 = self.avgpool(x)
        z  = torch.cat([y1, y2], 1) # concatenate
        z  = self.flatten(z)
        z  = self.linear(z)
        z  = F.log_softmax(z)
        return z



class DawnNet(nn.Module):
    def __init__(self):
        super(DawnNet, self).__init__()
        self.conv   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = nn.Sequential(*[
                        Block(inplanes=64,  outplanes=64,  stride=1),
                        Block(inplanes=64,  outplanes=64,  stride=1)])
        self.layer2 = nn.Sequential(*[
                        Block(inplanes=64,  outplanes=128, stride=2),
                        Block(inplanes=128, outplanes=128, stride=1)])
        self.layer3 = nn.Sequential(*[
                        Block(inplanes=128, outplanes=256, stride=2),
                        Block(inplanes=256, outplanes=256, stride=1)])
        self.layer4 = nn.Sequential(*[
                        Block(inplanes=256, outplanes=256, stride=2),
                        Block(inplanes=256, outplanes=256, stride=1)])
        self.final  = Final(inplanes=256, out_features=10)


    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)   # [N,3,32,32] -> [N,64,32,32]
        y = self.layer1(y) # output [N,  64, 32, 32]
        y = self.layer2(y) # output [N, 128, 16, 16]
        y = self.layer3(y) # output [N, 256,  8,  8]
        y = self.layer4(y) # output [N, 256,  4,  4]
        y = self.final(y)  # output [N, 10]
        return y

