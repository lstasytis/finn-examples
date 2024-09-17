# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import alveo_default_platform
from finn.builder.build_dataflow_config import default_build_dataflow_steps
from qonnx.transformation.insert_topk import InsertTopK
from finn.builder.build_dataflow_steps import build_dataflow_step_lookup
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
import os
import shutil
import numpy as np
import onnx
from onnx import helper as oh
import sys
import json


folding_method_choice = int(sys.argv[1])
output_type = int(sys.argv[2])
verify = int(sys.argv[3])
padding = int(sys.argv[4])
folding_dwc_heuristic = int(sys.argv[5])
auto_fifo_type = int(sys.argv[6])
folding_max_attempts = int(sys.argv[7])


if auto_fifo_type == 0:
    auto_fifo_depths = False
    auto_fifo_strategy = "characterize"
elif auto_fifo_type == 1:
    auto_fifo_depths = True
    auto_fifo_strategy = "characterize"
elif auto_fifo_type == 2:
    auto_fifo_depths = True
    auto_fifo_strategy = "characterize_analytic"



models = [
    "cnv_1w1a_gtsrb",
]

# which platforms to build the networks for
zynq_platforms = ["Pynq-Z1"]
platforms_to_build = zynq_platforms

folding_methods = ["default", "naive", "optimized", "optimized_same_throughput"]

if folding_method_choice not in [0,1,2]:
    print("folding method has to be set from range [1,2,3] (default, naive folding, optimized folding)")
else:
    exit
folding = folding_methods[folding_method_choice]




build_dir = os.environ["FINN_BUILD_DIR"]
os.chdir(f"{build_dir}/../projects/finn/finn-examples/build/gtsrb/")


if output_type == 0:
    build_steps = build_cfg.estimate_only_dataflow_steps
    generate_outputs = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS]   

elif output_type == 1:
    build_steps = build_cfg.stitch_only_build_dataflow_steps
    generate_outputs = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
                              build_cfg.DataflowOutputType.STITCHED_IP]
elif output_type == 2:
    build_steps = build_cfg.default_build_dataflow_steps
    generate_outputs = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
                            build_cfg.DataflowOutputType.STITCHED_IP,
                            build_cfg.DataflowOutputType.BITFILE,]

verification_steps = [
    build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON,
    build_cfg.VerificationStepType.TIDY_UP_PYTHON,
    build_cfg.VerificationStepType.STREAMLINED_PYTHON,
    build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
]

def custom_step_update_model(model, cfg):
    op = onnx.OperatorSetIdProto()
    op.version = 11
    load_model = onnx.load(model_file)
    update_model = onnx.helper.make_model(load_model.graph, opset_imports=[op])
    model_ref = ModelWrapper(update_model)
    # onnx.save(update_model, "models/%s_updated.onnx" % model_name)

    return model_ref

def custom_step_add_preproc(model, cfg):
    # GTSRB data with raw uint8 pixels is divided by 255 prior to training
    # reflect this in the inference graph so we can perform inference directly
    # on raw uint8 data
    in_name = model.graph.input[0].name
    new_in_name = model.make_new_valueinfo_name()
    new_param_name = model.make_new_valueinfo_name()
    div_param = np.asarray(255.0, dtype=np.float32)
    new_div = oh.make_node(
        "Div",
        [in_name, new_param_name],
        [new_in_name],
        name="PreprocDiv",
    )
    model.set_initializer(new_param_name, div_param)
    model.graph.node.insert(0, new_div)
    model.graph.node[1].input[0] = new_in_name
    # set input dtype to uint8
    model.set_tensor_datatype(in_name, DataType["UINT8"])
    return model

# Insert TopK node to get predicted Top-1 class
def step_preprocess(model, cfg):
    model = model.transform(InsertTopK(k=1))
    return model

build_dataflow_step_lookup["step_preprocess_InsertTopK"] = step_preprocess

custom_build_steps = (
    [custom_step_update_model]
    + [custom_step_add_preproc]
    + ["step_preprocess_InsertTopK"]
    + build_steps
)


# determine which shell flow to use for a given platform
def platform_to_shell(platform):
    if platform in zynq_platforms:
        return build_cfg.ShellFlowType.VIVADO_ZYNQ
    else:
        raise Exception("Unknown platform, can't determine ShellFlowType")


# create a release dir, used for finn-examples release packaging
os.makedirs("release", exist_ok=True)

for platform_name in platforms_to_build:
    shell_flow_type = platform_to_shell(platform_name)
    vitis_platform = None
    # for Zynq, use the board name as the release name
    # e.g. ZCU104
    release_platform_name = platform_name
    platform_dir = "release/%s" % release_platform_name
    os.makedirs(platform_dir, exist_ok=True)
    for model_name in models:



        # Set up variables needed for verifying build
        ci_folder = "../../ci"
        io_folder = ci_folder + "/verification_io"
        if os.getenv("VERIFICATION_EN", "0") in {"0", "1"}:
            shutil.copy(ci_folder + "/verification_funcs.py", ".")
            from verification_funcs import (
                create_logger,
                set_verif_steps,
                set_verif_io,
                verify_build_output,
            )

            create_logger()
            verif_steps = set_verif_steps()
            verif_input, verif_output = set_verif_io(io_folder, model_name)




        if folding == "default":
            folding_config_file="folding_config/%s_folding_config.json" % "gtsrb"
            target_fps=None  
            style="naive"     

        elif folding == "naive":
            folding_config_file = None

            test = f"gtsrb/output_{model_name}_{platform_name}"
            dir = f'{build_dir}/../projects/finn/finn-examples/build/{test}_default_0_0_2_1'
            with open(dir+f"/report/estimate_network_performance.json") as f:
                throughput_rep = json.load(f)
            target_fps=int(throughput_rep['estimated_throughput_fps'])
            style = "naive"

        elif folding == "optimized":
            folding_config_file = None
            test = f"gtsrb/output_{model_name}_{platform_name}"
            dir = f'{build_dir}/../projects/finn/finn-examples/build/{test}_default_0_0_2_1'
            with open(dir+f"/report/estimate_network_performance.json") as f:
                throughput_rep = json.load(f)
            target_fps=int(throughput_rep['estimated_throughput_fps'])
            style="optimizer"
            

        # set up the build configuration for this model
        cfg = build_cfg.DataflowBuildConfig(
            output_dir="output_%s_%s_%s_%s_%s_%s_%s" % (model_name, release_platform_name, folding, padding,folding_dwc_heuristic, auto_fifo_type,folding_max_attempts),
            steps=custom_build_steps,
            folding_config_file=folding_config_file,
            target_fps=target_fps,
            folding_style=style,
            folding_max_attempts=folding_max_attempts,
            folding_maximum_padding=padding,
            enable_folding_dwc_heuristic=folding_dwc_heuristic,
            auto_fifo_depths = auto_fifo_depths,
            auto_fifo_strategy = auto_fifo_strategy,
            synth_clk_period_ns=10.0,
            board=platform_name,
          #  verify_steps=verif_steps,
          #  verify_input_npy=verif_input,
          #  verify_expected_output_npy=verif_output,
          #  verify_save_full_context=True,

            shell_flow_type=shell_flow_type,
            vitis_platform=vitis_platform,
            save_intermediate_models=True,
            generate_outputs=[
                build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                build_cfg.DataflowOutputType.STITCHED_IP,
                build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
                build_cfg.DataflowOutputType.BITFILE,
                build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
                build_cfg.DataflowOutputType.PYNQ_DRIVER,
            ],
            specialize_layers_config_file="specialize_layers_config/gtsrb_specialize_layers.json",
        )
        model_file = "models/%s.onnx" % model_name
        # launch FINN compiler to build
        build.build_dataflow_cfg(model_file, cfg)

        if os.getenv("VERIFICATION_EN") == "1" and verify == 1:
            # Verify build using verification output
            verify_build_output(cfg, model_name)


        # copy bitfiles into release dir if found
        bitfile_gen_dir = cfg.output_dir + "/bitfile"
        files_to_check_and_copy = [
            "finn-accel.bit",
            "finn-accel.hwh",
            "finn-accel.xclbin",
        ]
        for f in files_to_check_and_copy:
            src_file = bitfile_gen_dir + "/" + f
            dst_file = platform_dir + "/" + f.replace("finn-accel", model_name)
            if os.path.isfile(src_file):
                shutil.copy(src_file, dst_file)


