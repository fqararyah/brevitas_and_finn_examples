import onnx
from finn.util.test import get_test_model_trained
import brevitas.onnx as bo
import settings 
from finn.util.visualization import showSrc, showInNetron
from finn.util.basic import make_build_dir
import os

settings_obj = settings.Settings('./settings.txt')
model_name = "TFC"

build_dir = settings_obj.out_dir + '/finn_out/' + model_name + '/build'

# if not os.path.exists(build_dir):
#     if not os.path.exists(settings_obj.out_dir + '/finn_out/' + model_name):
#         os.mkdir(settings_obj.out_dir + '/finn_out/' + model_name)
#     os.mkdir(build_dir)

tfc = get_test_model_trained(model_name, 1, 1)


bo.export_finn_onnx(tfc, (1, 1, 28, 28), build_dir+"/tfc_w1_a1.onnx")

#showInNetron(build_dir+"/tfc_w1_a1.onnx")

from finn.core.modelwrapper import ModelWrapper
model = ModelWrapper(build_dir+"/tfc_w1_a1.onnx")

from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.fold_constants import FoldConstants

model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(InferDataTypes())
model = model.transform(RemoveStaticGraphInputs())

model.save(build_dir+"/tfc_w1_a1_tidy.onnx")

#showInNetron(build_dir+"/tfc_w1_a1_tidy.onnx")

from finn.util.pytorch import ToTensor
from finn.transformation.merge_onnx_models import MergeONNXModels
from finn.core.datatype import DataType

model = ModelWrapper(build_dir+"/tfc_w1_a1_tidy.onnx")
global_inp_name = model.graph.input[0].name
ishape = model.get_tensor_shape(global_inp_name)
# preprocessing: torchvision's ToTensor divides uint8 inputs by 255
totensor_pyt = ToTensor()
chkpt_preproc_name = build_dir+"/tfc_w1_a1_preproc.onnx"
bo.export_finn_onnx(totensor_pyt, ishape, chkpt_preproc_name)

# join preprocessing and core model
pre_model = ModelWrapper(chkpt_preproc_name)
model = model.transform(MergeONNXModels(pre_model))
# add input quantization annotation: UINT8 for all BNN-PYNQ models
global_inp_name = model.graph.input[0].name
model.set_tensor_datatype(global_inp_name, DataType["UINT8"])

model.save(build_dir+"/tfc_w1_a1_with_preproc.onnx")
#showInNetron(build_dir+"/tfc_w1_a1_with_preproc.onnx")

from finn.transformation.insert_topk import InsertTopK

# postprocessing: insert Top-1 node at the end
model = model.transform(InsertTopK(k=1))
chkpt_name = build_dir+"/tfc_w1_a1_pre_post.onnx"
# tidy-up again
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(InferDataTypes())
model = model.transform(RemoveStaticGraphInputs())
model.save(chkpt_name)

#showInNetron(build_dir+"/tfc_w1_a1_pre_post.onnx")

from finn.transformation.streamline import Streamline
showSrc(Streamline)

from finn.transformation.streamline.reorder import MoveScalarLinearPastInvariants
import finn.transformation.streamline.absorb as absorb

model = ModelWrapper(build_dir+"/tfc_w1_a1_pre_post.onnx")
# move initial Mul (from preproc) past the Reshape
model = model.transform(MoveScalarLinearPastInvariants())
# streamline
model = model.transform(Streamline())
model.save(build_dir+"/tfc_w1_a1_streamlined.onnx")
#showInNetron(build_dir+"/tfc_w1_a1_streamlined.onnx")

from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.general import RemoveUnusedTensors

model = model.transform(ConvertBipolarMatMulToXnorPopcount())
model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
# absorb final add-mul nodes into TopK
model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
model = model.transform(RoundAndClipThresholds())

# bit of tidy-up
model = model.transform(InferDataLayouts())
model = model.transform(RemoveUnusedTensors())

model.save(build_dir+"/tfc_w1a1_ready_for_hls_conversion.onnx")
#showInNetron(build_dir+"/tfc_w1a1_ready_for_hls_conversion.onnx")



import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import shutil


rtlsim_output_dir = build_dir + '/rtl_sim/'

#Delete previous run results if exist
if os.path.exists(rtlsim_output_dir):
    shutil.rmtree(rtlsim_output_dir)
    print("Previous run results deleted!")

#replace_verilog_relpaths.ReplaceVerilogRelPaths()

# cfg_stitched_ip = build.DataflowBuildConfig(
#     output_dir          = rtlsim_output_dir,
#     target_fps          = 100000,
#     synth_clk_period_ns = 10.0,
#     #fpga_part           = "xc7z020clg400-1",
#     board                = "ZCU104",
#     steps               = build_cfg.estimate_only_dataflow_steps,
#     generate_outputs=[
#         build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
#         build_cfg.DataflowOutputType.STITCHED_IP,
#         build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
#         build_cfg.DataflowOutputType.OOC_SYNTH,
#     ]
# )

cfg_stitched_ip = build.DataflowBuildConfig(
    output_dir          = rtlsim_output_dir,
    target_fps          = 10000,
    synth_clk_period_ns = 10.0,
    fpga_part           = "xc7z020clg400-1",
    #board                = "ZCU104",
    #steps               = build_cfg.estimate_only_dataflow_steps,
    generate_outputs=[
    build_cfg.DataflowOutputType.STITCHED_IP,
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.OOC_SYNTH,
    build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
    build_cfg.DataflowOutputType.BITFILE,
    build_cfg.DataflowOutputType.PYNQ_DRIVER,
    build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ]
    )

build.build_dataflow_cfg(build_dir+"/tfc_w1a1_ready_for_hls_conversion.onnx", cfg_stitched_ip)

#from finn.util.visualization import showInNetron
#showInNetron(settings_obj.out_dir + 'finn_lenet.onnx')