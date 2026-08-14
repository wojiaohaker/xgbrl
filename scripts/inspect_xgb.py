"""Load XGB USD model in Isaac Sim for visual inspection.

Run with: ~/Softwares/IsaacLab/isaaclab.sh -p scripts/inspect_xgb.py --visualizer kit
"""

import argparse

from isaaclab.app import AppLauncher
from isaaclab_tasks.utils import add_launcher_args

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

# Create simulation
sim_cfg = sim_utils.SimulationCfg(device="cuda:0", dt=0.005)
sim = SimulationContext(sim_cfg)
sim.reset()

# Add ground plane
ground_cfg = sim_utils.GroundPlaneCfg()
ground_cfg.func("/World/ground", ground_cfg)

# Load the pre-converted USD file directly
usd_path = "/home/qiyuan/WorkSpace/IsaacSimWorkSpace/xgbdog/xgb.usd"
usd_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
usd_cfg.func("/World/Robot", usd_cfg, translation=(0.0, 0.0, 0.32))

# Run a few steps to let physics settle
for _ in range(100):
    sim.step(render=True)

print("=" * 60)
print(f"XGB USD model loaded from: {usd_path}")
print("Check if all 4 legs look symmetric in the viewport.")
print("Press Ctrl+C to exit.")
print("=" * 60)

try:
    while simulation_app.is_running():
        sim.step(render=True)
except KeyboardInterrupt:
    pass

simulation_app.close()
