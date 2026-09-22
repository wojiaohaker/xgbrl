一、训练命令

```
可视化调试
cd /home/qiyuan/Softwares/xgbrl && \
/home/qiyuan/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-XGB-v0 \
    --visualizer kit \
    --num_envs=64 \
    --max_iterations 50

正式训练
cd /home/qiyuan/Softwares/xgbrl && \
/home/qiyuan/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-XGB-v0 \
    --headless \
    --num_envs=4096 \
    --max_iterations 2000
```



二、播放训练

```
可视化调试
cd /home/qiyuan/Softwares/xgbrl && /home/qiyuan/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_rsl_rl.py \
  --task Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/xgb_flat/2026-09-22_13-59-16/model_1999.pt \
  --viz kit
```

