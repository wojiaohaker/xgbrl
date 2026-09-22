# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export trained RSL-RL policy as ONNX with node names matching qiyuan_mc.

Usage:
    ./isaaclab.sh -p -m xgbrl.export_onnx \
        --checkpoint /path/to/model_XXXX.pt \
        --output /path/to/policy.onnx

qiyuan_mc expects ONNX node names: input/h0/c0 → output/hn/cn
(RSL-RL default: obs/h_in/c_in → actions/h_out/c_out)
"""

import argparse
import os

import torch
import torch.nn as nn


class PolicyExporter(nn.Module):
    """Wrapper that runs LSTM + actor and exports with qiyuan_mc-compatible names."""

    def __init__(self, rnn, actor, normalizer=None):
        super().__init__()
        self.rnn = rnn
        self.actor = actor
        self.normalizer = normalizer if normalizer is not None else nn.Identity()

    def forward(self, x_in, h_in, c_in):
        x = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c


def main():
    parser = argparse.ArgumentParser(description="Export RSL-RL policy as ONNX for qiyuan_mc")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from rsl_rl.runners import OnPolicyRunner

    # Load checkpoint
    runner = OnPolicyRunner(args.checkpoint, continue_training=False)
    policy = runner.alg.actor_critic
    policy.cpu().eval()

    # Extract RNN and actor
    if hasattr(policy, "memory_a"):
        rnn = policy.memory_a.rnn
    elif hasattr(policy, "memory_s"):
        rnn = policy.memory_s.rnn
    else:
        raise ValueError("Policy has no RNN memory module")

    actor = policy.actor
    normalizer = getattr(policy, "normalizer", None)

    # Create exporter
    exporter = PolicyExporter(rnn, actor, normalizer)
    exporter.cpu().eval()

    # Export with qiyuan_mc-compatible node names
    obs_dim = rnn.input_size
    hidden_dim = rnn.hidden_size
    num_layers = rnn.num_layers

    dummy_obs = torch.zeros(1, obs_dim)
    dummy_h = torch.zeros(num_layers, 1, hidden_dim)
    dummy_c = torch.zeros(num_layers, 1, hidden_dim)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.onnx.export(
        exporter,
        (dummy_obs, dummy_h, dummy_c),
        args.output,
        export_params=True,
        opset_version=18,
        verbose=args.verbose,
        input_names=["input", "h0", "c0"],
        output_names=["output", "hn", "cn"],
        dynamic_axes={},
    )
    print(f"Exported policy ONNX: {args.output}")
    print(f"  input: ({obs_dim},)  h0: ({num_layers},1,{hidden_dim})  c0: ({num_layers},1,{hidden_dim})")
    print(f"  output: (1,12)  hn: ({num_layers},1,{hidden_dim})  cn: ({num_layers},1,{hidden_dim})")


if __name__ == "__main__":
    main()
