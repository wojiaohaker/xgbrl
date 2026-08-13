一、训练命令

```
~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-XGB-v0 --num_envs=4096
```



二、play

```
~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint logs/rsl_rl/xgb_flat/2026-08-13_18-03-55/model_299.pt \
  --visualizer kit
```



三、1w

````
先改迭代
训练命令：

```bash
cd ~/Softwares/xgbrl && ~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-XGB-v0 --num_envs=4096
```

`max_iterations` 已改为 10000。
````

