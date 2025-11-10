# Copyright (C) 2020-2022, Xilinx, Inc.
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
from qonnx.util.config import extract_model_config_to_json
from finn.util.basic import compute_total_model_fifo_size
from qonnx.core.modelwrapper import ModelWrapper
import time

# custom steps for mobilenetv1
from custom_steps import (
    step_mobilenet_streamline,
    step_mobilenet_convert_to_hw_layers,
    step_mobilenet_convert_to_hw_layers_separate_th,
    step_mobilenet_lower_convs,
    step_mobilenet_slr_floorplan,
)

model_name = "mobilenetv1-w4a4"
model_file = "models/%s_pre_post_tidy_opset-11.onnx" % model_name

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


# select target clock frequency
def select_clk_period(platform):
    if platform in zynq_platforms:
        return 5.4
    elif platform in alveo_platforms:
        return 3.0


# select build steps (ZCU104/102 folding config is based on separate thresholding nodes)
def select_build_steps(platform):
    if platform in zynq_platforms:
        return [
            step_mobilenet_streamline,
            step_mobilenet_lower_convs,
            step_mobilenet_convert_to_hw_layers_separate_th,
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_set_fifo_depths",
        ]
    elif platform in alveo_platforms:
        return [
            step_mobilenet_streamline,
            step_mobilenet_lower_convs,
            step_mobilenet_convert_to_hw_layers,
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_set_fifo_depths",
        ]


# create a release dir, used for finn-examples release packaging
os.makedirs("release", exist_ok=True)


# removing "largefifo_rtlsim" as it takes multiple days to run
# methods = ["analytic_model_based", "analytic_rtlsim", "manual-placement"]
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

        last_output_dir = "output_%s_%s" % (model_name, release_platform_name)
        cfg = build_cfg.DataflowBuildConfig(
            steps=select_build_steps(platform_name),
            output_dir=last_output_dir,
            folding_config_file="folding_config/%s_folding_config.json" % platform_name,
            synth_clk_period_ns=select_clk_period(platform_name),
            board=platform_name,
            shell_flow_type=shell_flow_type,
            vitis_platform=vitis_platform,
            # folding config comes with FIFO depths already
            auto_fifo_depths=auto_fifo_depths,
            auto_fifo_strategy=auto_fifo_strategy,
            tav_generation_strategy=tav_generation_strategy_key,
            # enable extra performance optimizations (physopt)
            vitis_opt_strategy=build_cfg.VitisOptStrategyCfg.PERFORMANCE_BEST,
            skip_resynth_during_fifo_sizing=True,
            generate_outputs=[
                build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            ],
            specialize_layers_config_file="specialize_layers_config/%s_specialize_layers.json"
            % platform_name,
        )

        # Build the model without verification
        t0 = time.time()
        build.build_dataflow_cfg(model_file, cfg)
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
        # from qonnx.custom_op.registry import getCustomOp
        # import pdb;breakpoint()
