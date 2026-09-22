# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export trained RSL-RL policy as ONNX with node names matching qiyuan_mc.

Usage:
    ./isaaclab.sh -p -m xgbrl.export_onnx \
        --checkpoint /path/to/model_XXXX.pt \
        --output /path/to/policy.onnx

qiyuan_mc expects ONNX node names: input/h0/c0 -> output/hn/cn
"""

import argparse
import os

import torch
import torch.nn as nn


class PolicyExporter(nn.Module):
    """Wrapper that runs LSTM + actor MLP and exports with qiyuan_mc-compatible names."""

    def __init__(self, lstm: nn.LSTM, mlp: nn.Sequential):
        super().__init__()
        self.lstm = lstm
        self.mlp = mlp

    def forward(self, x_in, h_in, c_in):
        x, (h, c) = self.lstm(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.mlp(x), h, c


def main():
    parser = argparse.ArgumentParser(description="Export RSL-RL policy as ONNX for qiyuan_mc")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    state = ckpt["actor_state_dict"]

    # Infer dims from checkpoint
    # rnn.rnn.weight_ih_l0: (4*hidden, input)
    input_dim = state["rnn.rnn.weight_ih_l0"].shape[1]
    hidden_dim = state["rnn.rnn.weight_hh_l0"].shape[1]
    num_layers = 1  # checkpoint only has l0 weights

    # Build LSTM
    lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
    lstm_weights = {
        "weight_ih_l0": state["rnn.rnn.weight_ih_l0"],
        "weight_hh_l0": state["rnn.rnn.weight_hh_l0"],
        "bias_ih_l0": state["rnn.rnn.bias_ih_l0"],
        "bias_hh_l0": state["rnn.rnn.bias_hh_l0"],
    }
    lstm.load_state_dict(lstm_weights)
    lstm.eval()

    # Build MLP (actor): hidden_dims = [128, 128], output = 12
    # MLP layers: Linear(input, 128), ELU, Linear(128, 128), ELU, Linear(128, 12)
    mlp_dims = []
    mlp_dims.append(state["mlp.0.weight"].shape[0])  # first hidden
    # find remaining hidden dims
    i = 2
    while f"mlp.{i}.weight" in state:
        if f"mlp.{i+2}.weight" in state:
            mlp_dims.append(state[f"mlp.{i}.weight"].shape[0])
        i += 2
    output_dim = state[f"mlp.{i}.weight"].shape[0] if f"mlp.{i}.weight" in state else state["mlp.4.weight"].shape[0]

    layers = []
    prev_dim = hidden_dim
    for dim in mlp_dims:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(nn.ELU())
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))
    mlp = nn.Sequential(*layers)

    # Load MLP weights
    mlp_state = {}
    for i in range(0, len(mlp_dims) * 2 + 1, 2):
        layer_idx = i
        mlp_state[f"{layer_idx}.weight"] = state[f"mlp.{layer_idx}.weight"]
        mlp_state[f"{layer_idx}.bias"] = state[f"mlp.{layer_idx}.bias"]
    mlp.load_state_dict(mlp_state)
    mlp.eval()

    # Create exporter
    exporter = PolicyExporter(lstm, mlp)
    exporter.cpu().eval()

    # Export with qiyuan_mc-compatible node names
    dummy_obs = torch.zeros(1, input_dim)
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
    print(f"  input: ({input_dim},)  h0: ({num_layers},1,{hidden_dim})  c0: ({num_layers},1,{hidden_dim})")
    print(f"  output: ({output_dim},)  hn: ({num_layers},1,{hidden_dim})  cn: ({num_layers},1,{hidden_dim})")


if __name__ == "__main__":
    main()
