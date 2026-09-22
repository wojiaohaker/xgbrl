一、训练命令

```
~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-XGB-v0 --num_envs=4096


cd ~/Softwares/xgbrl && ~/Softwares/IsaacLab/isaaclab.sh train --rl_library rsl_rl --task=Isaac-Velocity-Flat-XGB-v0 --num_envs=4096


IsaacLabTutorial
# 无头训练（推荐）
~/Softwares/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task Template-Isaac-Lab-Tutorial-Direct-v0 --headless

# 带可视化训练
~/Softwares/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task Template-Isaac-Lab-Tutorial-Direct-v0 --visualizer kit

# 指定训练轮数
~/Softwares/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task Template-Isaac-Lab-Tutorial-Direct-v0 --headless --max_iterations 1000

这个项目用的是 **skrl** 库（不是 rsl_rl），任务名是 `Template-Isaac-Lab-Tutorial-Direct-v0`。
```

```
cd /home/qiyuan/Softwares/xgbrl && \
/home/qiyuan/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-XGB-v0 \
    --visualizer kit \
    --num_envs=4096 \
    --max_iterations 2000
```

