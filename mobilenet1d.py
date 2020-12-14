from torch import nn
import torch
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
#import torchvision.models as models
#from torch.autograd import Variable


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.ReLU6(inplace=True)
        )


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, round_nearest=8):
        super(MobileNetV1, self).__init__()
        input_channel = 32
        last_channel = 1280

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv1d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm1d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv1d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm1d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv1d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool1d(1),
        )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(1, input_channel, stride=2)]
        

        #self.fc = nn.Linear(1024, 1000)

        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            #nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes[0]),
            nn.LogSoftmax(dim=1),)

        self.normalize = nn.BatchNorm1d(6420)
        self.maxpool1d = nn.MaxPool1d(3, stride=2)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if(len(x.shape) ==2):
            x = self.normalize(x)
            x = x.reshape([x.shape[0], 1, x.shape[1]])
            x = self.maxpool1d(x)
        #print(x.shape)
        x = self.features(x)
        x = x.mean(2)
        #x = x.view(-1, 1024)
        x = self.classifier(x)
        return x
