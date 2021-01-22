# A modified ResNet used for speeding up CIFAR10 training.
# Reference :
#      https://github.com/davidcpage/cifar10-fast
#      https://myrtle.ai/learn/how-to-train-your-resnet-1-baseline/
#      https://lambdalabs.com/blog/resnet9-train-to-94-cifar10-accuracy-in-100-seconds/

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(inplanes, outplanes, stride):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride,
            padding=1, bias=False)

def conv1x1(inplanes, outplanes, stride):
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride,
                            padding=0, bias=False)


def print_cuda_time(method, prefix_str=''):
    def wrapped_fn(*args, **kw):
        start = torch.cuda.Event(enable_timing=True)
        end  = torch.cuda.Event(enable_timing=True)
        start.record()

        result = method(*args, **kw)

        end.record()
        torch.cuda.synchronize()
        print(prefix_str, start.elapsed_time(end), 'ms')
        return result
    return wrapped_fn

"""
The modified Blocks from David Page
  https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
"""
class DawnBlock(nn.Module):
    def __init__(self, inplanes: int, outplanes: int, stride: int=1) -> None:
        print('DawnBlock.__init__')
        super(DawnBlock, self).__init__()
        self.downsample = (inplanes != outplanes) or (stride != 1)
        self.bn1    = nn.BatchNorm2d(inplanes)
        self.relu1  = nn.ReLU()
        self.c3x3a  = conv3x3(inplanes, outplanes, stride)
        self.bn2    = nn.BatchNorm2d(outplanes)
        self.relu2  = nn.ReLU()
        self.c3x3b  = conv3x3(outplanes, outplanes, stride=1)
        self.c1x1   = conv1x1(inplanes, outplanes, stride)

    def forward(self, x: Tensor) -> Tensor:
        y  = self.relu1(self.bn1(x))
        z  = self.relu2(self.bn2(self.c3x3a(y)))
        z  = self.c3x3b(z)
        z += self.c1x1(y) if self.downsample else y
        return z


'''
Hongyi Zhang, Yann N. Dauphin, Tengyu Ma. Fixup Initialization: Residual
Learning Without Normalization. ICLR 2019
Ref code for ResNet : https://github.com/hongyi-zhang/Fixup
WideResNet
https://github.com/ajbrock/BoilerPlate/blob/master/Models/fixup.py

Timing for fixup is slower compared to DawnBlock which uses BatchNorm.
'''
class FixupBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride) :

        print('FixupBlock.__init__')

        super(FixupBlock, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1  = conv3x3(inplanes, outplanes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu1  = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2  = conv3x3(outplanes, outplanes, stride=1)
        self.scale  = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = (inplanes != outplanes) or (stride != 1)
        self.relu2   = nn.ReLU(inplace=True)
        if self.downsample :
            self.c1x1   = conv1x1(inplanes, outplanes, stride)

        # Fixup init scales He init by L^(-1/(2m-2))
        k1, c1  = self.conv1.kernel_size, self.conv1.out_channels
        n1 = k1[0] * k1[1] * c1 # Fan in
        L, m = 4, 2  # number of residual blocks, layers in residual block
        s = L ** (1/(2 * m - 2)) # scaling factor for the normalization
        self.conv1.weight.data.normal_(0, s * math.sqrt(2. / n1))
        self.conv2.weight.data.zero_()
        if self.downsample:
            k2, c2 = self.c1x1.kernel_size, self.c1x1.out_channels
            n2 = k2[0] * k2[1] * c2
            self.c1x1.weight.data.normal_(0, math.sqrt(2. / n2))


    def forward(self, x):
        # identity = x
        # y = self.conv1(x + self.bias1a)
        # y = self.relu1(y + self.bias1b)
        # y = self.conv2(y + self.bias2a)
        # y = y * self.scale + self.bias2b
        # if self.downsample :
        #     identity = self.c1x1(x + self.bias1a)
        # y += identity
        # y = self.relu2(y)

        identity = x
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        if self.downsample :
            identity = self.c1x1(x)
        y += identity
        y = self.relu2(y)

        return y


class DawnFinal(nn.Module):
    def __init__(self, inplanes: int, out_features: int ):
        super(DawnFinal, self).__init__()
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
        z  = F.log_softmax(z, dim=1)
        return z


class FixupFinal(nn.Module):
    def __init__(self, inplanes: int, out_features: int ):
        super(FixupFinal, self).__init__()
        self.avgpool = nn.AvgPool2d(4) # [N, inplanes, H/4, W/4]
        self.maxpool = nn.MaxPool2d(4) # [N, inplanes, H/4, W/4]
        self.flatten = nn.Flatten()
        self.linear  = nn.Linear(2 * inplanes, out_features, bias=True)

        # Fixup init
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.maxpool(x)
        y2 = self.avgpool(x)
        z  = torch.cat([y1, y2], 1) # concatenate
        z  = self.flatten(z)
        z  = self.linear(z)
        z  = F.log_softmax(z, dim=1)
        return z

import time

class DawnNet(nn.Module):
    def __init__(self, Block, Final):
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


        self.count = 0
        self.debug_time = 0



    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)   # [N,3,32,32] -> [N,64,32,32]
        y = self.layer1(y) # output [N,  64, 32, 32]
        y = self.layer2(y) # output [N, 128, 16, 16]
        y = self.layer3(y) # output [N, 256,  8,  8]
        y = self.layer4(y) # output [N, 256,  4,  4]
        y = self.final(y)  # output [N, 10]
        return y

def profile_model_autograd():
    import torch
    # model = DawnNet(DawnBlock, DawnFinal).cuda()
    model = DawnNet(FixupBlock, FixupFinal).cuda()
    x = torch.randn((512,3,32,32), requires_grad=True).cuda()
    y = torch.ones(512, dtype=torch.long).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Half precision
    model.half()
    for layer in model.modules():
        layer.float()

    timed_forward  = print_cuda_time(lambda: model(x), 'forward:')
    timed_backward = print_cuda_time(lambda: loss.backward(), 'backward:' )

    @print_cuda_time
    def fn() :
        for _ in range(5):
            optimizer.zero_grad()
            log_prob = model(x)
            loss = F.nll_loss(log_prob, y)
            loss.backward()
            optimizer.step()


    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        fn()
        # for _ in range(5):
        #     optimizer.zero_grad()
        #     log_prob = model(x)
        #     loss = F.nll_loss(log_prob, y)
        #     loss.backward()
        #     optimizer.step()



    # print(prof)

def profile_model_torchprof():
    """https://github.com/awwong1/torchprof"""
    import torch
    import torchprof
    model = DawnNet(DawnBlock, DawnFinal).cuda()
    # model = DawnNet(FixupBlock, FixupFinal).cuda()

    # Half precision
    model.half()
    for layer in model.modules():
        layer.float()

    x = torch.rand((1,3,32,32)).cuda()
    with torchprof.Profile(model, use_cuda=True, profile_memory=True) as prof:
        for _ in range(10):
            model(x)

    print(prof.display(show_events=False))


if __name__ == "__main__":
    profile_model_autograd()
    # profile_model_torchprof()

    print('End of profiling')

