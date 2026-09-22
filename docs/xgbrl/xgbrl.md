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

