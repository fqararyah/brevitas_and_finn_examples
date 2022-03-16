"""
Modified from: https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models

MIT License

Copyright (c) 2019 Xilinx, Inc (Alessandro Pappalardo)
Copyright (c) 2018 Oleg Sémery

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


__all__ = ['quant_mobilenet_v1']
import torch
from torch import nn, resolve_neg, save
from torch.nn import Sequential

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantAvgPool2d
from brevitas.quant import IntBias

from common import CommonIntActQuant, CommonUintActQuant
from common import CommonIntWeightPerChannelQuant, CommonIntWeightPerTensorQuant
import configparser
import numpy as np

from brevitas_examples.imagenet_classification.models import model_with_cfg

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

FIRST_LAYER_BIT_WIDTH = 8


class DwsConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            bit_width,
            pw_activation_scaling_per_channel=False,
            save_activations = False):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            weight_bit_width=bit_width,
            act_bit_width=bit_width,
            save_activations=save_activations)
        self.pw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            weight_bit_width=bit_width,
            act_bit_width=bit_width,
            activation_scaling_per_channel=pw_activation_scaling_per_channel,
            save_activations=save_activations)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            weight_bit_width,
            act_bit_width,
            stride=1,
            padding=0,
            groups=1,
            bn_eps=1e-5,
            activation_scaling_per_channel=False,
            save_activations = False):
        super(ConvBlock, self).__init__()
        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width)
        self.save_activations = save_activations
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            per_channel_broadcastable_shape=(1, out_channels, 1, 1),
            scaling_per_channel=activation_scaling_per_channel,
            return_quant_tensor=True)

    def forward(self, x):
        x = self.conv(x)
        if self.save_activations:
            save_activations(x, 'conv')
        x = self.bn(x)
        if self.save_activations:
            save_activations(x, 'bn')
        x = self.activation(x)
        if self.save_activations:
            save_activations(x, 'relu')
        return x


class MobileNet(nn.Module):

    def __init__(
            self,
            channels,
            first_stage_stride,
            bit_width,
            in_channels=3,
            num_classes=1000,
            save_activations = False):
        super(MobileNet, self).__init__()
        init_block_channels = channels[0][0]

        self.features = Sequential()
        init_block = ConvBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=3,
            stride=2,
            weight_bit_width=FIRST_LAYER_BIT_WIDTH,
            activation_scaling_per_channel=True,
            act_bit_width=bit_width,
            save_activations = save_activations)
        self.features.add_module('init_block', init_block)
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels[1:]):
            stage = Sequential()
            pw_activation_scaling_per_channel = i < len(channels[1:]) - 1
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                mod = DwsConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bit_width=bit_width,
                    pw_activation_scaling_per_channel=pw_activation_scaling_per_channel,
                    save_activations=save_activations)
                stage.add_module('unit{}'.format(j + 1), mod)
                in_channels = out_channels
            self.features.add_module('stage{}'.format(i + 1), stage)
        self.final_pool = QuantAvgPool2d(kernel_size=7, stride=1, bit_width=bit_width)
        self.output = QuantLinear(
            in_channels, num_classes,
            bias=True,
            bias_quant=IntBias,
            weight_quant=CommonIntWeightPerTensorQuant,
            weight_bit_width=bit_width)

    def forward(self, x):
        x = self.features(x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        out = self.output(x)
        return out

layer_rank = 0
save_path = '/home/fareed/wd/models/weights_activations/brevitas/mobilenetv1'
def save_activations(activations , layer_type, file_path = save_path):
    global layer_rank 
    with open(file_path + '/' + layer_type + '_' + str(layer_rank) + '.txt', 'w') as f:
        actiovations_npy = activations[0].detach().numpy()
        #print(activations[0])
        activations_shape = actiovations_npy.shape
        for i in range(activations_shape[0]):
            for j in range(activations_shape[1]):
                for k in range(activations_shape[2]):
                    f.write(str(actiovations_npy[i][j][k]) + ' ')
                f.write('\n')
            f.write('\n')

    layer_rank += 1

def save_weights(model, file_path = save_path):
    indices_conv_layers = {}
    indx = 0
    for layer_name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d): 
            #print(layer.quant_weight().zero_point.item())
            #print(layer.quant_weight().scale.size())
            weights_npy = layer.quant_weight().value
            scale = layer.quant_weight().scale
            zero_point = layer.quant_weight().zero_point.item()
            with open(file_path + '/_weights' + layer_name + '.txt', 'w') as f:
                weights_shape = weights_npy.shape
                print(weights_shape)
                for filter in range(weights_shape[0]):
                    for i in range(weights_shape[1]):
                        for j in range(weights_shape[2]):
                            for k in range(weights_shape[3]):
                                f.write(str( round( weights_npy[filter][i][j][k].item() / scale[filter].item() + zero_point) ) + ' ')
                            f.write('\n')
                        f.write('\n')
        
        indx += 1

def quant_mobilenet_v1(cfg, save_activations = False):

    channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
    first_stage_stride = False
    width_scale = float(cfg.get('MODEL', 'WIDTH_SCALE'))
    bit_width = cfg.getint('QUANT', 'BIT_WIDTH')

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    net = MobileNet(
        channels=channels,
        first_stage_stride=first_stage_stride,
        bit_width=bit_width,
        save_activations=save_activations)

    return net

config_file = '/home/fareed/wd/my_repos/brevitas_and_finn/brevitas_models/cfg/quant_mobilenet_v1_4b.ini'
config = configparser.ConfigParser()
config.read(config_file)
#option1
#model = quant_mobilenet_v1(config, save_activations = True)

#option2
model, cfg = model_with_cfg('/home/fareed/wd/my_repos/brevitas_and_finn/brevitas_models/cfg/quant_mobilenet_v1_4b',\
     True)

#option3
#from finn.util.test import get_test_model
#model = get_test_model(netname = "mobilenet", wbits = 4, abits = 4, pretrained = True)

img_np = np.random.randint(low= 0, high = 256, size=(3, 224, 224))
np.save(save_path + "/_mobilenet_input.npy", img_np)
img_np = img_np.reshape(1, 3, 224, 224)
img_torch = torch.from_numpy(img_np).float()

model.eval()
save_weights(model)
#model.forward(img_torch)

