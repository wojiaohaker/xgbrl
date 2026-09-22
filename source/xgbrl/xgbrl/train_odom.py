# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train odom estimation network (supervised) for qiyuan_mc deployment.

The odom network estimates body-frame linear velocity from proprioception:
    Input  (29-dim): (roll, pitch, jpos_delta[12], jvel×0.05[12], gyro×0.25[3])
    Output (3-dim):  base_lin_vel_b (body-frame linear velocity)

Data is collected by running a trained policy in IsaacLab, then the LSTM
network is trained with MSE loss. The exported ONNX matches qiyuan_mc's
expected node names: input/h0/c0 → output/hn/cn.

Usage:
    # Step 1: Collect data + train (requires trained policy checkpoint)
    ./isaaclab.sh -p -m xgbrl.train_odom \
        --checkpoint /path/to/model_XXXX.pt \
        --task Isaac-Velocity-Flat-XGB-Play-v0 \
        --num_envs 50 \
        --num_steps 5000 \
        --output /path/to/odom.onnx

    # Step 2 (after training): Export ONNX separately
    ./isaaclab.sh -p -m xgbrl.train_odom \
        --export_only \
        --odom_pth /path/to/odom.pth \
        --output /path/to/odom.onnx
"""

import argparse
import os
import torch
import torch.nn as nn


class OdomNet(nn.Module):
    """LSTM(512) + MLP → 3, matching deployment odom_mix_walk architecture."""

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
        """x: (1, batch, 29), h_in/c_in: (1, batch, 512) → output: (batch, 3)"""
        x, (h, c) = self.lstm(x, (h_in, c_in))
        x = x.squeeze(0)
        return self.mlp(x), h, c


def collect_data(args):
    """Run trained policy in IsaacLab, collect proprioception + GT base_vel."""
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    sim_app = app_launcher.app

    import gymnasium as gym
    import xgbrl.tasks  # noqa: F401 - registers XGB envs
    from rsl_rl.runners import OnPolicyRunner

    # Create environment
    env = gym.make(args.task, num_envs=args.num_envs, device="cuda:0")
    env = env.unwrapped

    # Load trained policy
    runner = OnPolicyRunner(args.checkpoint, env, continue_training=False, device="cuda:0")
    policy = runner.alg.actor_critic
    policy.eval()

    robot = env.scene["robot"]

    # Data storage
    all_inputs = []   # (N, 29)
    all_targets = []  # (N, 3)

    obs, _ = env.reset()
    # Get LSTM initial states
    h = torch.zeros(1, args.num_envs, 512, device="cuda:0")
    c = torch.zeros(1, args.num_envs, 512, device="cuda:0")

    print(f"Collecting {args.num_steps} steps across {args.num_envs} envs...")

    with torch.no_grad():
        for step in range(args.num_steps):
            # Run policy
            obs_tensor = torch.tensor(obs["policy"], device="cuda:0", dtype=torch.float32)
            if obs_tensor.dim() == 2:
                obs_tensor = obs_tensor.unsqueeze(0)  # (1, batch, 48)
            action, h, c = policy(obs_tensor, h, c)
            action = action.squeeze(0) if action.dim() == 3 else action

            # Step environment
            obs, _, terminated, truncated, _ = env.step(action)

            # Extract proprioception for odom training
            # roll, pitch from quaternion
            qw = robot.data.root_quat_w[:, 0]
            qx = robot.data.root_quat_w[:, 1]
            qy = robot.data.root_quat_w[:, 2]
            qz = robot.data.root_quat_w[:, 3]
            roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
            pitch = torch.asin(torch.clamp(2 * (qw * qy - qz * qx), -1, 1))

            # jpos_delta = joint_pos - default_joint_pos
            jpos_delta = robot.data.joint_pos - robot.data.default_joint_pos

            # jvel × 0.05
            jvel_scaled = robot.data.joint_vel * 0.05

            # gyro × 0.25 (body angular velocity)
            gyro_scaled = robot.data.root_ang_vel_b * 0.25

            # GT target: body-frame linear velocity
            base_vel = robot.data.root_lin_vel_b

            # Build 29-dim input: (roll, pitch, jpos_delta[12], jvel_scaled[12], gyro_scaled[3])
            odom_input = torch.cat([
                roll.unsqueeze(1),       # (N, 1)
                pitch.unsqueeze(1),      # (N, 1)
                jpos_delta,              # (N, 12)
                jvel_scaled,             # (N, 12)
                gyro_scaled,             # (N, 3)
            ], dim=1)  # (N, 29)

            all_inputs.append(odom_input.cpu())
            all_targets.append(base_vel.cpu())

            if (step + 1) % 500 == 0:
                print(f"  Step {step+1}/{args.num_steps}")

            # Reset LSTM state for reset environments
            reset = terminated | truncated
            if reset.any():
                h[:, reset, :] = 0
                c[:, reset, :] = 0

    env.close()
    sim_app.close()

    inputs = torch.cat(all_inputs, dim=0)   # (N*num_envs, 29)
    targets = torch.cat(all_targets, dim=0)  # (N*num_envs, 3)
    print(f"Collected {inputs.shape[0]} samples")
    return inputs, targets


def train_odom(inputs, targets, args):
    """Train LSTM odom network with MSE loss on sequential data."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_samples = inputs.shape[0]
    n_envs = args.num_envs
    n_steps = n_samples // n_envs

    # Reshape to (n_steps, n_envs, 29) for sequential processing
    inputs_seq = inputs[:n_steps * n_envs].reshape(n_steps, n_envs, 29).to(device)
    targets_seq = targets[:n_steps * n_envs].reshape(n_steps, n_envs, 3).to(device)

    model = OdomNet(input_dim=29, hidden_dim=512, output_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    seq_len = args.seq_len  # BPTT length
    n_batches = n_steps // seq_len

    print(f"Training: {n_steps} steps × {n_envs} envs, seq_len={seq_len}, {n_batches} batches/epoch")

    for epoch in range(args.epochs):
        total_loss = 0
        h = torch.zeros(1, n_envs, 512, device=device)
        c = torch.zeros(1, n_envs, 512, device=device)

        for i in range(n_batches):
            start = i * seq_len
            end = start + seq_len
            x = inputs_seq[start:end]  # (seq_len, n_envs, 29)
            y = targets_seq[start:end]  # (seq_len, n_envs, 3)

            # Detach hidden state between batches (truncated BPTT)
            h = h.detach()
            c = c.detach()

            preds = []
            for t in range(seq_len):
                out, h, c = model(x[t:t+1], h, c)  # (1, n_envs, 3)
                preds.append(out)

            preds = torch.stack(preds).squeeze(1)  # (seq_len, n_envs, 3)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.6f}")

    # Save checkpoint
    pth_path = args.output.replace(".onnx", ".pth")
    torch.save(model.state_dict(), pth_path)
    print(f"Saved odom weights: {pth_path}")
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


def main():
    parser = argparse.ArgumentParser(description="Train odom estimation network")
    parser.add_argument("--checkpoint", help="Path to trained policy .pt checkpoint")
    parser.add_argument("--task", default="Isaac-Velocity-Flat-XGB-Play-v0")
    parser.add_argument("--num_envs", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=64, help="BPTT sequence length")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--export_only", action="store_true", help="Skip training, just export")
    parser.add_argument("--odom_pth", help="Path to saved odom .pth for export_only mode")
    args = parser.parse_args()

    if args.export_only:
        model = OdomNet()
        model.load_state_dict(torch.load(args.odom_pth, map_location="cpu"))
        export_onnx(model, args.output)
        return

    # Collect data
    inputs, targets = collect_data(args)

    # Train
    model = train_odom(inputs, targets, args)

    # Export
    export_onnx(model, args.output)


if __name__ == "__main__":
    main()
