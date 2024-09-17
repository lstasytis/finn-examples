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
import os
import shutil
import sys
import json

folding_method_choice = int(sys.argv[1])
output_type = int(sys.argv[2])
verify = int(sys.argv[3])
padding = int(sys.argv[4])
folding_dwc_heuristic = int(sys.argv[5])
auto_fifo_type = int(sys.argv[6])
folding_max_attempts = int(sys.argv[7])


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

# the BNN-PYNQ models -- these all come as exported .onnx models
# see models/download_bnn_pynq_models.sh
models = [
    #"tfc-w1a1",
   # "tfc-w1a2",
   # "tfc-w2a2",
    "cnv-w1a1",
    #"cnv-w1a2",
   # "cnv-w2a2",
]

# which platforms to build the networks for
#zynq_platforms = ["Pynq-Z1", "Ultra96", "ZCU104"]
#alveo_platforms = ["U250"]
zynq_platforms = ["Ultra96"]
alveo_platforms = []
platforms_to_build = zynq_platforms + alveo_platforms

folding_methods = ["default", "naive", "optimized", "optimized_same_throughput"]





build_dir = os.environ["FINN_BUILD_DIR"]
os.chdir(f"{build_dir}/../projects/finn/finn-examples/build/bnn-pynq/")



if folding_method_choice not in [0,1,2]:
    print("folding method has to be set from range [1,2,3] (default, naive folding, optimized folding)")
else:
    exit
folding = folding_methods[folding_method_choice]

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
else:
    build_steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    "step_convert_to_hw",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_set_fifo_depths",
    "step_generate_estimate_reports", 

 #   "step_hw_codegen",
  #  "step_hw_ipgen",
  #  "step_synth_ip",
  #  "step_create_stitched_ip",

]

    generate_outputs = [build_cfg.DataflowOutputType.BITFILE,]
    #model_file = "/scratch/users/lstasyti/projects/finn/finn-examples/build/kws/output_MLP_W3A3_python_speech_features_pre-processing_QONNX_Pynq-Z1_optimized/intermediate_models/step_hw_ipgen.onnx"


if auto_fifo_type == 0:
    auto_fifo_depths = False
    auto_fifo_strategy = "characterize"
elif auto_fifo_type == 1:
    auto_fifo_depths = True
    auto_fifo_strategy = "characterize"
elif auto_fifo_type == 2:
    auto_fifo_depths = True
    auto_fifo_strategy = "characterize_analytic"


# determine which shell flow to use for a given platform
def platform_to_shell(platform):
    if platform in zynq_platforms:
        return build_cfg.ShellFlowType.VIVADO_ZYNQ
    elif platform in alveo_platforms:
        return build_cfg.ShellFlowType.VITIS_ALVEO
    else:
        raise Exception("Unknown platform, can't determine ShellFlowType")




# create a release dir, used for finn-examples release packaging
os.makedirs("release", exist_ok=True)

for platform_name in platforms_to_build:
    shell_flow_type = platform_to_shell(platform_name)
    if shell_flow_type == build_cfg.ShellFlowType.VITIS_ALVEO:
        vitis_platform = alveo_default_platform[platform_name]
        # for Alveo, use the Vitis platform name as the release name
        # e.g. xilinx_u250_xdma_201830_2
        release_platform_name = vitis_platform
    else:
        vitis_platform = None
        # for Zynq, use the board name as the release name
        # e.g. ZCU104
        release_platform_name = platform_name
    platform_dir = "release/%s" % release_platform_name
    os.makedirs(platform_dir, exist_ok=True)


    for model_name in models:

        if folding == "default":
            folding_config_file="folding_config/%s_folding_config.json" % model_name
            target_fps=None  
            style="naive"     

        elif folding == "naive":
            folding_config_file = None

            test = f"bnn-pynq/output_{model_name}_{platform_name}"
            dir = f'{build_dir}/../projects/finn/finn-examples/build/{test}_default_0_0_2_1'
            with open(dir+f"/report/estimate_network_performance.json") as f:
                throughput_rep = json.load(f)
            target_fps=int(throughput_rep['estimated_throughput_fps'])
            style = "naive"

        elif folding == "optimized":
            folding_config_file = None
            test = f"bnn-pynq/output_{model_name}_{platform_name}"
            dir = f'{build_dir}/../projects/finn/finn-examples/build/{test}_default_0_0_2_1'
            with open(dir+f"/report/estimate_network_performance.json") as f:
                throughput_rep = json.load(f)
            target_fps=int(throughput_rep['estimated_throughput_fps'])
            style="optimizer"
            


        if "tfc" in model_name:
            verif_input, verif_output = set_verif_io(io_folder, "tfc_mnist")
        elif "cnv" in model_name:
            verif_input, verif_output = set_verif_io(io_folder, "cnv_cifar10")

        # set up the build configuration for this model
        cfg = build_cfg.DataflowBuildConfig(
            output_dir="output_%s_%s_%s_%s_%s_%s_%s" % (model_name, release_platform_name, folding, padding,folding_dwc_heuristic, auto_fifo_type,folding_max_attempts),
            steps=build_steps,
            folding_config_file=folding_config_file,
            target_fps=target_fps,
            folding_style=style,
            folding_max_attempts=folding_max_attempts,
            folding_maximum_padding=padding,
            enable_folding_dwc_heuristic=folding_dwc_heuristic,
            auto_fifo_depths = auto_fifo_depths,
            auto_fifo_strategy = auto_fifo_strategy,
            verify_steps=verif_steps,
            verify_input_npy=verif_input,
            verify_expected_output_npy=verif_output,
            verify_save_full_context=True,

            synth_clk_period_ns=5.0,
            board=platform_name,
            shell_flow_type=shell_flow_type,
            vitis_platform=vitis_platform,
            generate_outputs=generate_outputs,
            save_intermediate_models=True,
            default_swg_exception=True,
            specialize_layers_config_file="specialize_layers_config/%s_specialize_layers.json"
            % model_name,
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
