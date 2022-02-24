

import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant
from brevitas.export import FINNManager
from torch.nn import Module

import sys
import brevitas.onnx as bo
import numpy as np
import torch
sys.path.append('/home/fareed/wd/finn/src/')
sys.path.append('/home/fareed/wd/qonnx/src')

from finn.core.modelwrapper import ModelWrapper

import settings 
import finn.transformation.fpgadataflow.replace_verilog_relpaths as replace_verilog_relpaths

class LowPrecisionLeNet(Module):
    def __init__(self):
        super(LowPrecisionLeNet, self).__init__()
        self.quant_inp = qnn.QuantIdentity(
            bit_width=4, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(
            3, 6, 5, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu1 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(
            6, 16, 5, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.fc1   = qnn.QuantLinear(
            400, 120, bias=True, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.fc2   = qnn.QuantLinear(
            120, 84, bias=True, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu4 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.fc3   = qnn.QuantLinear(
            84, 10, bias=False, weight_bit_width=3)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(out))
        print(out.size())
        out = F.max_pool2d(out, 2)
        print(out.size())
        out = self.relu2(self.conv2(out))
        print(out.size())
        out = F.max_pool2d(out, 2)
        print(out.size())
        out = out.reshape(out.shape[0], -1)
        print(out.size())
        out = self.relu3(self.fc1(out))
        print(out.size())
        out = self.relu4(self.fc2(out))
        print(out.size())
        out = self.fc3(out)
        print(out.size())
        return out


low_precision_lenet = LowPrecisionLeNet()
img_torch = torch.from_numpy(np.random.randint(low=0, high=256, size=[1,3,32,32])).float()

#low_precision_lenet.forward(img_torch)
#exit()
# ... training ...

settings_obj = settings.Settings('./settings.txt')
model_file_name = settings_obj.out_dir + 'brevitas_out/finn_lenet.onnx'
processed_model_file_name = settings_obj.out_dir + 'brevitas_out/p_finn_lenet.onnx'

#model_for_sim = ModelWrapper(low_precision_lenet)

bo.export_finn_onnx(low_precision_lenet, (1, 3, 32, 32), model_file_name)

from finn.util.test import get_test_model_trained

model = ModelWrapper(model_file_name)
tfc = get_test_model_trained("mobilenet", 4, 4)
#for name, param in tfc.named_parameters():
#    print(name, param.size())
#exit()

model.save(processed_model_file_name)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import shutil


rtlsim_output_dir = settings_obj.out_dir + '/finn_out'

#Delete previous run results if exist
if os.path.exists(rtlsim_output_dir):
    shutil.rmtree(rtlsim_output_dir)
    print("Previous run results deleted!")

#replace_verilog_relpaths.ReplaceVerilogRelPaths()

cfg_stitched_ip = build.DataflowBuildConfig(
    output_dir          = rtlsim_output_dir,
    #target_fps          = 100000,
    synth_clk_period_ns = 10.0,
    fpga_part           = "xc7z020clg400-1",
    #board                = "ZCU104",
    steps               = build_cfg.estimate_only_dataflow_steps,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.OOC_SYNTH,
    ]
)
build.build_dataflow_cfg(processed_model_file_name, cfg_stitched_ip)

#from finn.util.visualization import showInNetron
#showInNetron(settings_obj.out_dir + 'finn_lenet.onnx')