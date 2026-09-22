# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train odom estimation network (supervised) for qiyuan_mc deployment.

The odom network estimates body-frame linear velocity from proprioception:
    Input  (29-dim): (roll, pitch, jpos_delta[12], jvel*0.05[12], gyro*0.25[3])
    Output (3-dim):  base_lin_vel_b (body-frame linear velocity)

Data is collected by running a trained policy in IsaacLab, then the LSTM
network is trained with MSE loss. The exported ONNX matches qiyuan_mc's
expected node names: input/h0/c0 -> output/hn/cn.

Usage:
    ./isaaclab.sh -p /home/qiyuan/Softwares/xgbrl/source/xgbrl/xgbrl/train_odom.py \
        --task Isaac-Velocity-Flat-XGB-Play-v0 \
        --checkpoint /path/to/model_XXXX.pt \
        --num_envs 50 --num_steps 5000 \
        --output /path/to/odom_mix_walk.onnx
"""

import argparse
import os

import torch
import torch.nn as nn

# CLI
parser = argparse.ArgumentParser(description="Train odom estimation network for qiyuan_mc")
parser.add_argument("--task", default="Isaac-Velocity-Flat-XGB-Play-v0")
parser.add_argument("--checkpoint", required=True, help="Path to trained policy .pt checkpoint")
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--num_steps", type=int, default=5000)
parser.add_argument("--seq_len", type=int, default=64, help="BPTT sequence length")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--output", required=True, help="Output .onnx path")
args_cli, remaining_args = parser.parse_known_args()

# AppLauncher (must be before IsaacLab imports)
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, args_remaining=remaining_args)
sim_app = app_launcher.app

# IsaacLab imports
import gymnasium as gym
import xgbrl.tasks  # noqa: F401 - registers XGB envs


# ---------------------------------------------------------------------------
# Policy model (matches RSL-RL RNNModel architecture)
# ---------------------------------------------------------------------------

class PolicyLSTM(nn.Module):
    """LSTM + MLP actor, reconstructed from RSL-RL checkpoint."""

    def __init__(self, input_dim, hidden_dim, mlp_dims, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1)
        layers = []
        prev = hidden_dim
        for dim in mlp_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ELU())
            prev = dim
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, h, c):
        """x: (batch, input_dim) -> output: (batch, output_dim), (h, c) updated."""
        x, (h, c) = self.lstm(x.unsqueeze(0), (h, c))
        return self.mlp(x.squeeze(0)), h, c


def load_policy_from_checkpoint(ckpt_path, device="cuda:0"):
    """Build PolicyLSTM from RSL-RL checkpoint."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    state = ckpt["actor_state_dict"]

    input_dim = state["rnn.rnn.weight_ih_l0"].shape[1]
    hidden_dim = state["rnn.rnn.weight_hh_l0"].shape[1]

    # Infer MLP dims from checkpoint
    mlp_dims = []
    i = 0
    while f"mlp.{i}.weight" in state:
        mlp_dims.append(state[f"mlp.{i}.weight"].shape[0])
        i += 2
    output_dim = mlp_dims.pop()  # last entry is output_dim

    model = PolicyLSTM(input_dim, hidden_dim, mlp_dims, output_dim)

    # Load LSTM weights
    lstm_state = {
        "weight_ih_l0": state["rnn.rnn.weight_ih_l0"],
        "weight_hh_l0": state["rnn.rnn.weight_hh_l0"],
        "bias_ih_l0": state["rnn.rnn.bias_ih_l0"],
        "bias_hh_l0": state["rnn.rnn.bias_hh_l0"],
    }
    model.lstm.load_state_dict(lstm_state)

    # Load MLP weights
    mlp_state = {}
    for i in range(0, (len(mlp_dims) + 1) * 2, 2):
        mlp_state[f"{i}.weight"] = state[f"mlp.{i}.weight"]
        mlp_state[f"{i}.bias"] = state[f"mlp.{i}.bias"]
    model.mlp.load_state_dict(mlp_state)

    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Odom network
# ---------------------------------------------------------------------------

class OdomNet(nn.Module):
    """LSTM(512) + MLP -> 3, matching deployment odom_mix_walk architecture."""

    def __init__(self, input_dim=29, hidden_dim=512, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x, h_in, c_in):
        """x: (1, batch, 29), h_in/c_in: (1, batch, 512) -> output: (batch, 3)"""
        x, (h, c) = self.lstm(x, (h_in, c_in))
        return self.mlp(x.squeeze(0)), h, c


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_data(env, policy_model, num_steps, num_envs, device):
    """Run trained policy, collect proprioception + GT base_vel."""
    robot = env.scene["robot"]
    all_inputs = []
    all_targets = []

    obs, _ = env.reset()
    h = torch.zeros(1, num_envs, 512, device=device)
    c = torch.zeros(1, num_envs, 512, device=device)

    print(f"Collecting {num_steps} steps across {num_envs} envs...")

    with torch.no_grad():
        for step in range(num_steps):
            obs_t = torch.tensor(obs["policy"], device=device, dtype=torch.float32)
            action, h, c = policy_model(obs_t, h, c)
            action = action.clamp(-1.0, 1.0)

            obs, _, terminated, truncated, _ = env.step(action)

            # Extract proprioception for odom training
            qw = robot.data.root_quat_w[:, 0]
            qx = robot.data.root_quat_w[:, 1]
            qy = robot.data.root_quat_w[:, 2]
            qz = robot.data.root_quat_w[:, 3]
            roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
            pitch = torch.asin(torch.clamp(2 * (qw * qy - qz * qx), -1, 1))

            jpos_delta = robot.data.joint_pos - robot.data.default_joint_pos
            jvel_scaled = robot.data.joint_vel * 0.05
            gyro_scaled = robot.data.root_ang_vel_b * 0.25
            base_vel = robot.data.root_lin_vel_b

            odom_input = torch.cat([
                roll.unsqueeze(1), pitch.unsqueeze(1),
                jpos_delta, jvel_scaled, gyro_scaled,
            ], dim=1)

            all_inputs.append(odom_input.cpu())
            all_targets.append(base_vel.cpu())

            if (step + 1) % 500 == 0:
                print(f"  Step {step+1}/{num_steps}")

            reset = terminated | truncated
            if reset.any():
                h[:, reset, :] = 0
                c[:, reset, :] = 0

    inputs = torch.cat(all_inputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    print(f"Collected {inputs.shape[0]} samples")
    return inputs, targets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_odom(inputs, targets, num_envs, seq_len, epochs, lr):
    """Train LSTM odom network with MSE loss on sequential data."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_samples = inputs.shape[0]
    n_steps = n_samples // num_envs

    inputs_seq = inputs[:n_steps * num_envs].reshape(n_steps, num_envs, 29).to(device)
    targets_seq = targets[:n_steps * num_envs].reshape(n_steps, num_envs, 3).to(device)

    model = OdomNet(input_dim=29, hidden_dim=512, output_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_batches = n_steps // seq_len
    print(f"Training: {n_steps} steps x {num_envs} envs, seq_len={seq_len}, {n_batches} batches/epoch")

    for epoch in range(epochs):
        total_loss = 0
        h = torch.zeros(1, num_envs, 512, device=device)
        c = torch.zeros(1, num_envs, 512, device=device)

        for i in range(n_batches):
            start = i * seq_len
            x = inputs_seq[start:start + seq_len]
            y = targets_seq[start:start + seq_len]

            h = h.detach()
            c = c.detach()

            preds = []
            for t in range(seq_len):
                out, h, c = model(x[t:t + 1], h, c)
                preds.append(out)

            preds = torch.stack(preds).squeeze(1)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

    return model


def export_onnx(model, output_path):
    """Export odom network as ONNX with qiyuan_mc-compatible node names."""
    model.cpu().eval()
    dummy_input = torch.zeros(1, 1, 29)
    dummy_h = torch.zeros(1, 1, 512)
    dummy_c = torch.zeros(1, 1, 512)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_input, dummy_h, dummy_c),
        output_path,
        export_params=True,
        opset_version=18,
        input_names=["input", "h0", "c0"],
        output_names=["output", "hn", "cn"],
        dynamic_axes={},
    )
    print(f"Exported odom ONNX: {output_path}")
    print(f"  input: (1,1,29)  h0: (1,1,512)  c0: (1,1,512)")
    print(f"  output: (1,3)  hn: (1,1,512)  cn: (1,1,512)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda:0"

    # Load env config directly from the registered entry point
    from xgbrl.tasks.manager_based.xgbrl.flat_env_cfg import XgbFlatEnvCfg_PLAY

    env_cfg = XgbFlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = device

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = env.unwrapped

    # Load policy from checkpoint
    policy_model = load_policy_from_checkpoint(args_cli.checkpoint, device)

    # Collect data
    inputs, targets = collect_data(env, policy_model, args_cli.num_steps, args_cli.num_envs, device)

    env.close()

    # Train odom network (before sim_app.close() which terminates the process)
    model = train_odom(inputs, targets, args_cli.num_envs, args_cli.seq_len, args_cli.epochs, args_cli.lr)

    # Export ONNX
    export_onnx(model, args_cli.output)

    sim_app.close()


if __name__ == "__main__":
    main()
