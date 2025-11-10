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
from finn.util.basic import compute_total_model_fifo_size
from qonnx.core.modelwrapper import ModelWrapper
import time
from qonnx.util.config import extract_model_config_to_json

# the BNN-PYNQ models -- these all come as exported .onnx models
# see models/download_bnn_pynq_models.sh
models = [
    "tfc-w1a1",
    # "tfc-w1a2",
    # "tfc-w2a2",
    "cnv-w1a1",
    # "cnv-w1a2",
    # "cnv-w2a2",
]

verif_en = os.getenv("VERIFICATION_EN", "0")

# which platforms to build the networks for
zynq_platforms = ["ZCU104"]
alveo_platforms = []
platforms_to_build = zynq_platforms + alveo_platforms


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


# assemble build flow from custom and pre-existing steps
select_build_steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    "step_convert_to_hw",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_set_fifo_depths",
    # "step_hw_codegen",
    # "step_hw_ipgen",
    # "step_create_stitched_ip",
    # "step_measure_rtlsim_performance",
    # "step_out_of_context_synthesis",
    # "step_synthesize_bitfile",
    # "step_make_driver",
    # "step_deployment_package",
]


methods = ["analytic_model_based"]


for platform_name in platforms_to_build:
    for method in methods:
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

        if method == "analytic_model_based":
            auto_fifo_strategy = "analytical"
            tav_generation_strategy_key = "tree_model"
            auto_fifo_depths = True
        elif method == "analytic_rtlsim":
            auto_fifo_strategy = "analytical"
            tav_generation_strategy_key = "rtlsim"
            auto_fifo_depths = True
        elif method == "largefifo_rtlsim":
            auto_fifo_strategy = "largefifo_rtlsim"
            tav_generation_strategy_key = "rtlsim"
            auto_fifo_depths = True
        else:
            auto_fifo_depths = False

        for model_name in models:
            # set up the build configuration for this model
            last_output_dir = "output_%s_%s" % (model_name, release_platform_name)
            cfg = build_cfg.DataflowBuildConfig(
                output_dir=last_output_dir,
                folding_config_file="folding_config/%s_folding_config.json" % model_name,
                synth_clk_period_ns=5.0,
                steps=select_build_steps,
                board=platform_name,
                auto_fifo_depths=auto_fifo_depths,
                auto_fifo_strategy=auto_fifo_strategy,
                tav_generation_strategy=tav_generation_strategy_key,
                shell_flow_type=shell_flow_type,
                vitis_platform=vitis_platform,
                skip_resynth_during_fifo_sizing=True,
                generate_outputs=[],
                save_intermediate_models=True,
                default_swg_exception=True,
                specialize_layers_config_file="specialize_layers_config/%s_specialize_layers.json"
                % model_name,
            )
            model_file = "models/%s.onnx" % model_name

            # Build the model without verification
            t0 = time.time()
            build.build_dataflow_cfg(model, cfg)
            t1 = time.time()

            model = ModelWrapper(last_output_dir + "/intermediate_models/step_set_fifo_depths.onnx")
            size, depth = compute_total_model_fifo_size(model)
            print(
                f"=================================\nfifo sizing method: {method}, size: {size // 1024//8}KB, depth: {depth}, time: {t1-t0}s"
            )

            hw_attrs = [
                "PE",
                "SIMD",
                "parallel_window",
                "ram_style",
                "depth",
                "impl_style",
                "resType",
                "mem_mode",
                "runtime_writeable_weights",
                "inFIFODepths",
                "outFIFODepths",
                "depth_trigger_uram",
                "depth_trigger_bram",
            ]

            extract_model_config_to_json(model, f"{model_name}_{method}.json", hw_attrs)
