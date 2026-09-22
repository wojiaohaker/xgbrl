一、训练模型

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



三、训练质量

```
cd /home/qiyuan/Softwares/xgbrl && /home/qiyuan/Softwares/IsaacLab/isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/xgb_flat --port 6006

浏览器打开 http://localhost:6006
```



四、导出训练

```
策略模型
cd /home/qiyuan/Softwares/xgbrl && /home/qiyuan/Softwares/IsaacLab/isaaclab.sh -p -m xgbrl.export_onnx \
  --checkpoint logs/rsl_rl/xgb_flat/2026-09-22_13-59-16/model_1999.pt \
  --output exported_models/policy_mix_walk.onnx
  
里程计模型
cd /home/qiyuan/Softwares/xgbrl && /home/qiyuan/Softwares/IsaacLab/isaaclab.sh -p source/xgbrl/xgbrl/train_odom.py \
  --task Isaac-Velocity-Flat-XGB-Play-v0 \
  --checkpoint logs/rsl_rl/xgb_flat/2026-09-22_13-59-16/model_1999.pt \
  --num_envs 50 --num_steps 5000 --epochs 50 \
  --output exported_models/odom_mix_walk.onnx
```



五、部署模型

```
# 1. 拷贝新模型（policy + odom，各含 .data） 
cp /home/qiyuan/Softwares/xgbrl/exported_models/policy_mix_walk.onnx \
   /home/qiyuan/Softwares/xgbrl/exported_models/policy_mix_walk.onnx.data \
   /home/qiyuan/Softwares/xgbrl/exported_models/odom_mix_walk.onnx \
   /home/qiyuan/Softwares/xgbrl/exported_models/odom_mix_walk.onnx.data \
   /home/qiyuan/Softwares/qiyuan_mc/models_xgbrl/ && echo "copy done"
```



六、qiyuan_mc测试

0、启动键盘脚本

```
cd /home/qiyuan/UnrealEngine/CarlaUE5

sudo python3 /home/qiyuan/UnrealEngine/CarlaUE5/Unreal/CarlaUnreal/Plugins/MuJoCoUE/Scripts/keyboard_control.py
```

1、mujoco_sim_matrix  原版matrix运控robot_mc

```
cd /home/qiyuan/Softwares/Mujoco330/mujoco_sim && \
./build/mujoco_sim config.yaml > /home/qiyuan/Softwares/qiyuan_mc/data/robot_mc/mujoco_sim_matrix.log 2>&1



cd /home/qiyuan/Softwares/Matrix/src/robot_mc/build/export/mc/bin && \
HOOK_OBS_LOG=/home/qiyuan/Softwares/qiyuan_mc/data/robot_mc/matrix_obs_log.txt \
HOOK_DUMP_DIR=/home/qiyuan/Softwares/qiyuan_mc/data/robot_mc/matrix_models_decrypted \
LD_PRELOAD=/home/qiyuan/Softwares/qiyuan_mc/tools/onnx_hook/libonnx_dump_hook.so \
HOOK_POL_FULL=1 HOOK_NO_FORENSICS=1 ROBOT_TYPE=XG \
./mc_ctrl r > /home/qiyuan/Softwares/qiyuan_mc/data/robot_mc/mc_ctrl_matrix.log 2>&1
```

2、mujoco_sim_qiyuan 自己写的运控qiyuan_mc + 复用matrix的加密onnx模型

```
cd /home/qiyuan/Softwares/Mujoco330/mujoco_sim && \
./build/mujoco_sim config.yaml > /home/qiyuan/Softwares/qiyuan_mc/data/qiyuan_mc/mujoco_sim_qiyuan.log 2>&1

cd /home/qiyuan/Softwares/qiyuan_mc/build && cmake --build . --target mc_ctrl -j$(nproc)

cd /home/qiyuan/Softwares/qiyuan_mc
./scripts/run_mc.sh XG 2>&1 | tee /home/qiyuan/Softwares/qiyuan_mc/data/qiyuan_mc/mc_ctrl_qiyuan.log 2>&1 
```

3、mujoco_sim_qiyuan 自己写的运控qiyuan_mc + 自己训练的xgbrl模型

```
cd /home/qiyuan/Softwares/Mujoco330/mujoco_sim && \
./build/mujoco_sim config.yaml > /home/qiyuan/Softwares/qiyuan_mc/data/qiyuan_mc_xgbrl/mujoco_sim_qiyuan_xgbrl.log 2>&1

cd /home/qiyuan/Softwares/qiyuan_mc/build && cmake --build . --target mc_ctrl -j$(nproc)

cd /home/qiyuan/Softwares/qiyuan_mc
./scripts/run_mc.sh XG xgbrl 2>&1 | tee /home/qiyuan/Softwares/qiyuan_mc/data/qiyuan_mc_xgbrl/mc_ctrl_qiyuan_xgbrl.log 2>&1 
```

