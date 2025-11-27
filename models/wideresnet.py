# note: we have implemented this WideResNet architechture with some help from https://github.com/xternalz/WideResNet-pytorch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    '''
    Residual block for the WideResNet consist of two stacked 
    Batch Normalization -> ReLu -> 3x3 Conv layer and 
    1x1 conv layer to match the dimensions of input and output channels
    '''
    def __init__(self, input_channels, output_channels, stride):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # checking input and output channel dimention
        # if input and output channel dimention doesn't match we use 1x1 conv 
        self.check_equal_in_out = (input_channels == output_channels)

        if not self.check_equal_in_out:
            self.shortcut = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = None
        

    def forward(self, x):
        if self.check_equal_in_out:
            out = F.relu(self.bn1(x), inplace=True)
            residual = x
        else:
            x = F.relu(self.bn1(x), inplace=True)
            out = x
            residual = self.shortcut(x)

        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True) 
        out = self.conv2(out)

        return torch.add(residual, out)


class WideResNet(nn.Module):
    '''
    Building WideResNet 
    '''
    def __init__(self, depth=28, widen_factor=2, num_classes=10):
        super(WideResNet, self).__init__()
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6

        # initilizing basic block
        block = BasicBlock

        # 1st conv layer before any residual network block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        # three groups of residual blocks
        # 1st block
        self.block1 = self._make_layer(block=block, input_channels=16, output_channels=16*widen_factor, nb_layers=n, stride=1)
        # 2nd block
        self.block2 = self._make_layer(block=block, input_channels=16*widen_factor, output_channels=32*widen_factor, nb_layers=n, stride=2)
        # 3rd block
        self.block3 = self._make_layer(block=block, input_channels=32*widen_factor, output_channels=64*widen_factor, nb_layers=n, stride=2)
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(64*widen_factor)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64*widen_factor, num_classes)
        self.nChannels = 64*widen_factor


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    
    def _make_layer(self, block, input_channels, output_channels, nb_layers, stride):
        '''
        creates residual blocks.
        '''
        layers = []
        for i in range(int(nb_layers)):
            in_channels = input_channels if i == 0 else output_channels
            stride = stride if i == 0 else 1
            layers.append(block(in_channels, output_channels, stride))
        return nn.Sequential(*layers)
