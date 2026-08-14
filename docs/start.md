启动环境

使用uv

```bash
source ~/Softwares/IsaacLab/env_isaaclab/bin/activate
```



训练

### 方法 1：训练时直接可视化（Kit 可视化器）

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=16 --max_iterations=100 \
  --viz kit
```

这会弹出 **Isaac Sim UI**，你能看到机器人在动。

> 注意：`--headless` 已弃用，用 `--viz kit` 代替。



### 方法 2：轻量可视化（Newton 可视化器）

如果 Kit 太卡或显示有问题，用 Newton 可视化器（轻量 OpenGL）：

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=16 --max_iterations=100 \
  --viz newton
```

Newton 可视化器支持：

- **WASD/箭头键** 移动视角
- **鼠标拖拽** 环顾
- **滚轮** 缩放
- **H** 隐藏/显示 UI
- **ESC** 退出



训练了什么？



让我帮你详细分析这次训练做了什么：


## 这次训练做了什么（详细解释）

### 一、训练命令解析

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=16 --max_iterations=100 \
  --viz newton
```

| 参数                                | 含义                           |
| ----------------------------------- | ------------------------------ |
| `--task=Isaac-Velocity-Flat-XGB-v0` | 训练 xgb 在平坦地面上行走      |
| `--num_envs=16`                     | 同时运行 16 个并行环境         |
| `--max_iterations=100`              | 训练 100 轮迭代                |
| `--viz newton`                      | 用 Newton 可视化器显示实时画面 |

---

### 二、训练目标

机器人要学习**速度跟踪任务**：

```
命令：vx=1.5 m/s, vy=0, wz=0  →  机器人：向前快走
命令：vx=0, vy=0.5 m/s, wz=0  →  机器人：向左平移
命令：vx=0, vy=0, wz=1.0 rad/s → 机器人：原地右转
```

---

### 三、奖励函数（机器人怎么得分）

| 奖励项                 | 权重    | 作用                       |
| ---------------------- | ------- | -------------------------- |
| `track_lin_vel_xy_exp` | +1.5    | ✅ 跟踪线速度（向前/左右）  |
| `track_ang_vel_z_exp`  | +0.75   | ✅ 跟踪角速度（旋转）       |
| `flat_orientation_l2`  | -2.5    | ❌ 惩罚倾斜（保持身体水平） |
| `lin_vel_z_l2`         | -2.0    | ❌ 惩罚上下跳动             |
| `ang_vel_xy_l2`        | -0.05   | ❌ 惩罚前后/左右翻滚        |
| `action_rate_l2`       | -0.01   | ❌ 惩罚动作变化太快（平滑） |
| `dof_torques_l2`       | -0.0002 | ❌ 惩罚用力矩太大（节能）   |
| `dof_acc_l2`           | -2.5e-7 | ❌ 惩罚关节加速度太大       |
| `feet_air_time`        | None    | ⚠️ 已禁用（无脚部传感器）   |

---

### 四、终止条件（什么时候算失败）

| 条件           | 触发                           |
| -------------- | ------------------------------ |
| `base_contact` | 基座（`base_link`）触地 → 摔倒 |
| `time_out`     | 达到最大步数（正常结束）       |

---

### 五、速度命令范围（机器人要学多快）

| 方向       | 范围             | 对齐 Matrix |
| ---------- | ---------------- | ----------- |
| 前后（vx） | 0 ~ 3.0 m/s      | ✅           |
| 左右（vy） | -1.0 ~ 1.0 m/s   | ✅           |
| 旋转（wz） | -3.0 ~ 3.0 rad/s | ✅           |

---

### 六、训练过程（100 轮迭代）

```
┌─────────────────────────────────────────────────────────────────┐
│  每次迭代（共 100 次）                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 收集数据：16 个环境 × 24 步 = 384 条经验                     │
│     ┌──────────────────────────────────────────────────────┐    │
│     │  环境 1: 机器人收到 vx=1.5 命令                       │    │
│     │         观测：当前速度、关节角度、命令...              │    │
│     │         动作：输出 12 个关节目标位置                   │    │
│     │         奖励：速度跟踪越好，得分越高                   │    │
│     │         新观测：执行动作后的状态                       │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                  │
│  2. PPO 算法更新神经网络                                         │
│     - 计算哪些动作好（优势函数）                                 │
│     - 增加好动作的概率                                           │
│     - 减少坏动作的概率                                           │
│                                                                  │
│  3. 记录日志：reward、episode_length、各种指标                   │
│                                                                  │
│  4. 重复 100 次                                                  │
│                                                                  │
─────────────────────────────────────────────────────────────────┘
```

---

### 七、训练输出

```
logs/rsl_rl/xgb_flat/2026-08-11_17-02-19/
├── model_0.pt      ← 初始模型（随机权重）
├── model_50.pt     ← 第 50 轮模型
├── model_99.pt     ← 最终模型（训练 100 轮）
├── events.out...   ← TensorBoard 日志（训练曲线）
└── params/
    ├── env.yaml    ← 环境配置
    └── agent.yaml  ← PPO 算法配置
```

---

### 八、100 次迭代够吗？

**不够。** 这只是验证配置能跑通。

| 迭代次数  | 预期效果                     |
| --------- | ---------------------------- |
| 100       | 机器人可能还站不稳，经常摔倒 |
| 300~500   | 能站立、缓慢行走             |
| 1000~3000 | 稳定行走、速度跟踪           |

---

### 九、下一步

```bash
# 正式训练（1000 次迭代）
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=1000 \
  --viz newton --max_visible_envs 4
```

训练完后用 TensorBoard 查看曲线：

```bash
tensorboard --logdir ~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/
```

然后浏览器打开 `http://localhost:6006` 看 reward 是否上升。



导出onnx



用 `play` 命令会自动导出 ONNX：

```bash
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint /home/qiyuan/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/2026-08-11_17-02-19/model_99.pt
```

运行后会自动生成：

```
logs/rsl_rl/xgb_flat/2026-08-11_17-02-19/
└── exported/
    ├── policy.pt      ← JIT 格式
    └── policy.onnx    ← ONNX 格式 ✅
```

---

**注意**：把 `2026-08-11_17-02-19` 换成你实际的训练目录时间戳。如果不确定，先查看：

```bash
ls ~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/
```

然后用最新的目录里的 `model_99.pt`（或 `model_49.pt`）。



mujoco中使用训练好的onnx模型控制xgb机械狗

好的，我来帮你创建一个 Python 脚本，在 MuJoCo 中使用训练好的 ONNX 模型控制 xgb 机器人。


现在运行脚本测试：


**脚本运行成功！** 🎉

机器人已经在 MuJoCo 中移动了：

```
[Step 50]  pos=(0.38, 0.12, 0.60)   ← 起始位置，高度 0.6m
[Step 100] pos=(1.94, 0.90, 0.31)   ← 向前移动，高度降到 0.31m
[Step 700] pos=(-3.40, 2.52, 0.13)  ← 继续移动，但有些漂移
```

---

## 使用方法

```bash
cd ~/Softwares/Mujoco330/demo_xgb

# 运行（使用默认 ONNX 模型）
~/Softwares/IsaacLab/env_isaaclab/bin/python run_onnx_policy.py

# 指定其他 ONNX 模型
~/Softwares/IsaacLab/env_isaaclab/bin/python run_onnx_policy.py /path/to/policy.onnx
```

---

## 当前状态分析

机器人**在动**但**不太稳定**：
- ✅ ONNX 模型加载成功
- ✅ 观测构造正确（48 维）
- ✅ PD 控制工作（Kp=20, Kd=0.7）
- ⚠️ 机器人有些漂移（没有直线前进）

**原因**：
1. **训练不足**：只训练了 100 次迭代，模型还很"笨"
2. **观测构造**：可能需要微调（如速度估计、归一化等）

---

## 下一步建议

### 1. 训练更多迭代

```bash
# 训练 1000 次迭代
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=1000
```

### 2. 用新模型测试

训练完后重新导出 ONNX，再运行脚本：

```bash
# 导出新 ONNX
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint /path/to/new_model.pt

# 测试新模型
~/Softwares/IsaacLab/env_isaaclab/bin/python run_onnx_policy.py
```

### 3. 调整速度命令

修改脚本中的 `VELOCITY_CMD` 变量：

```python
# 向前 1.0 m/s
VELOCITY_CMD = np.array([1.0, 0.0, 0.0])

# 向左平移
VELOCITY_CMD = np.array([0.0, 0.5, 0.0])

# 原地旋转
VELOCITY_CMD = np.array([0.0, 0.0, 1.0])
```

---

## 脚本文件位置

```
~/Softwares/Mujoco330/demo_xgb/run_onnx_policy.py
```

你可以随时修改这个脚本来调整：
- PD 参数（KP, KD）
- 默认关节位置（DEFAULT_JOINT_POS）
- 速度命令（VELOCITY_CMD）
- 控制频率（CONTROL_DT）



再次训练

```
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=16 --max_iterations=100 \
  --viz newton

# 现在跑这个，去喝杯咖啡
cd ~/Softwares/IsaacLab

./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=3000 \
  --viz newton
  
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=3000
```



用 `play` 命令导出 ONNX：

```bash
# 1. 先找到最新训练目录
ls ~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/

# 2. 导出 ONNX（替换成实际目录名）
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint /home/qiyuan/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/<最新目录>/model_2999.pt
```

导出后 ONNX 文件在：

```
logs/rsl_rl/xgb_flat/<最新目录>/exported/policy.onnx
```

---

**快速操作**：先查目录名，再导出：

```bash
# 一行搞定
LATEST=$(ls -t ~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/ | head -1) && \
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint /home/qiyuan/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/$LATEST/model_2999.pt
```



```
cd ~/Softwares/Mujoco330/demo_xgb

~/Softwares/IsaacLab/env_isaaclab/bin/python run_onnx_policy.py
```



**对的，这是正常的。**

`play.py` 的工作流程：

```
1. 加载模型权重          ← 你看到的部分
2. 导出 ONNX（瞬间完成）  ← 已经完成了！
3. 进入仿真循环           ← 现在卡在这一步（实时仿真）
```

ONNX 导出只需要 **torch.onnx.export**，对一个 128×3 层的小 MLP 来说，几毫秒就完成了。

---

## 验证 ONNX 是否生成

按 **Ctrl+C** 终止 play 命令，然后检查：

```bash
ls ~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/2026-08-12_09-55-03/exported/
```

应该看到：

```
exported/
├── policy.pt
├── policy.onnx
└── policy.onnx.data
```

---

## 然后在 MuJoCo 中测试

```bash
cd ~/Softwares/Mujoco330/demo_xgb

~/Softwares/IsaacLab/env_isaaclab/bin/python run_onnx_policy.py \
  ~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/2026-08-12_09-55-03/exported/policy.onnx
```

3000 次迭代的模型应该比之前 100 次的好很多——机器人应该能**站立并尝试行走**，而不是疯狂跳舞了。



在isaac 看训练的效果



可以！用 `play` 命令在 Isaac Sim 里可视化：

```bash
cd ~/Softwares/IsaacLab

./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint /home/qiyuan/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/2026-08-12_09-55-03/model_2999.pt \
  --viz kit
```

---

## 参数说明

| 参数                                     | 说明                               |
| ---------------------------------------- | ---------------------------------- |
| `--task=Isaac-Velocity-Flat-XGB-Play-v0` | Play 环境（单机器人，无随机化）    |
| `--num_envs=1`                           | 只渲染 1 个机器人                  |
| `--checkpoint`                           | 指定训练好的模型                   |
| `--viz kit`                              | 使用 Isaac Sim UI 可视化（高保真） |

---

## 预期效果

如果模型训练成功，你应该看到：
- 机器人**站立并尝试行走**（vx=0.5 m/s）
- 高度稳定在 0.32m 左右
- 有节奏的步态

如果还是乱飞，说明：
1. **训练没收敛** → 需要更多迭代或调整奖励函数
2. **观测空间不匹配** → MuJoCo 脚本和 Isaac Lab 不一致

---

## 快速验证

先运行 play 看效果，如果 Isaac Sim 里也乱飞，那就是**训练本身的问题**，不是 MuJoCo 集成的问题。



使用项目模板

完成！现在检查文件是否需要调整导入路径：


有问题！`xgbrl_env_cfg.py` 导入了 `.rough_env_cfg`，但新项目中没有这个文件。需要复制：


文件都就位了。现在安装项目并验证：


任务注册了，但名字是模板默认的 `Template-Xgbrl-v0`。需要检查 `__init__.py` 的注册方式：


任务注册成功（警告是因为 Isaac Lab 原有 XGB 配置被覆盖了，不影响使用）。

现在可以训练了：

```bash
cd ~/Softwares/xgbrl

~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-XGB-v0 --num_envs=4096
```

或者用模板生成的脚本：

```bash
cd ~/Softwares/xgbrl

~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train_rsl_rl.py --task=Isaac-Velocity-Flat-XGB-v0 --num_envs=4096
```

要我帮你开始训练吗？





**不影响训练质量。**

可视化器只是**渲染画面**，不参与物理仿真和 RL 算法计算。

---

## 可视化 vs 训练

| 组件                      | 作用         | 卡不卡影响什么     |
| ------------------------- | ------------ | ------------------ |
| **可视化器** (Newton/Kit) | 画画面给你看 | 只影响你看到的帧率 |
| **物理引擎** (PhysX)      | 计算力学仿真 | ✅ 影响训练质量     |
| **RL 算法** (PPO)         | 更新神经网络 | ✅ 影响训练质量     |

**可视化卡 = 画面帧率低，但物理和算法照常跑。**

---

## 但可视化卡会拖慢训练速度

因为 `--viz newton` 会**等渲染完成**才继续下一步。

| 模式                  | 训练速度           |
| --------------------- | ------------------ |
| `--viz newton`        | 慢（要渲染）       |
| `--viz kit`           | 更慢（高保真渲染） |
| 无 `--viz` (headless) | **最快**           |

---

## 建议：训练时去掉可视化

```bash
# 正式训练（无渲染，最快）
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=3000
```

训练完后用 `play` 看效果：

```bash
# 可视化播放（1 个环境，不卡）
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint logs/rsl_rl/xgb_flat/<最新目录>/model_2999.pt \
  --viz newton
```

---

## 总结

| 问题                     | 答案                            |
| ------------------------ | ------------------------------- |
| 可视化卡影响训练质量吗？ | ❌ 不影响                        |
| 可视化卡影响训练速度吗？ | ✅ 会拖慢                        |
| 怎么解决？               | 训练时去掉 `--viz`，play 时再加 |



一、安装conda

你的系统没有安装 conda。有两个选择：

## 方案 1：安装 Miniforge（推荐）

```bash
# 下载 Miniforge
清华镜像
wget https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
# 安装完后重启终端或：
source ~/.bashrc

# 然后继续
conda create -n isaaclab python=3.11
conda activate isaaclab
```



二、Isaac Sim 与 Isaac Lab

## Isaac Sim 与 Isaac Lab 的关系

```
Isaac Lab（RL/机器人框架）
    ↑ 构建在
Isaac Sim（物理仿真引擎 + 渲染）
    ↑ 构建在
Omniverse / USD（NVIDIA 平台）
```

|          | Isaac Sim                        | Isaac Lab                       |
| -------- | -------------------------------- | ------------------------------- |
| **定位** | 仿真平台（引擎）                 | 机器人 RL 框架（上层）          |
| **功能** | 物理仿真、渲染、传感器、场景编辑 | RL 环境定义、训练接口、任务管理 |
| **类比** | 相当于 MuJoCo                    | 相当于 gymnasium + legged_gym   |
| **谁用** | 你不直接碰它                     | 你写训练代码时用它              |
| **安装** | 先装 Isaac Sim                   | 再装 Isaac Lab（依赖 Sim）      |

**简单理解**：
- **Isaac Sim** = 提供物理世界（重力、碰撞、渲染、相机）
- **Isaac Lab** = 在 Sim 上面封装了 RL 训练接口（观测/动作/奖励/并行环境管理）

你训练机器人时，代码是写在 Isaac Lab 层的，但它底层调用 Isaac Sim 来跑物理仿真。两者版本必须匹配，所以安装时要一起装对应版本组合。





好问题。官方组合没问题，**问题出在 conda**。

## 根因

| 安装方式                    | libstdc++ 来源                                  | 结果              |
| --------------------------- | ----------------------------------------------- | ----------------- |
| **conda-forge Python 3.11** | conda 自带 GCC 13 的 libstdc++（CXXABI 1.3.15） | 和 Isaac Sim 冲突 |
| **系统 Python 3.11 + venv** | 系统 GCC 11 的 libstdc++（CXXABI 1.3.14）       | 完全兼容          |
| **Omniverse Launcher**      | 自带完整隔离环境                                | 完全兼容          |
| **Docker 官方镜像**         | NVIDIA 精心匹配的库版本                         | 完全兼容          |

官方说的"Python 3.11"指的是**系统安装的 Python 3.11**（和系统 libstdc++ 一致），不是 conda-forge 编译的 Python 3.11（自带新版 libstdc++）。

## 修复方案：用 venv 替代 conda

```bash
# 安装系统 Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# 创建 venv（用系统 Python 3.11，libstdc++ 一致）
python3.11 -m venv ~/Softwares/isaaclab_venv
source ~/Softwares/isaaclab_venv/bin/activate

# 装 Isaac Sim 5.1.0
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com

# 装 Isaac Lab 2.3.2
cd ~/Softwares/IsaacLab
git checkout v2.3.2
pip install -e source/isaaclab
./isaaclab.sh --install rsl_rl
```

这样 Python 3.11 和 Isaac Sim 都用系统 libstdc++，不会有 ABI 冲突。

**你想试这个方案，还是直接降级 4.5.0？**



四、确认环境

逐步验证：

**1. 确认 Isaac Sim 5.1.0**

```bash

pip show isaacsim-core | grep Version
```

Isaac Sim 导入成功了！只是没有 `__version__` 属性。换个方式查版本：

```bash
pip show isaacsim-core | grep Version
```

然后继续验证 Isaac Lab：

```bash
python -c "import isaaclab; print('Isaac Lab OK')"
python -c "from rsl_rl.runners import OnPolicyRunner; print('RSL_RL OK')"
```



(isaaclab_venv) qiyuan@qiyuan-ue:~/Softwares/IsaacLab$ pip show isaacsim-core | grep Version
Version: 5.1.0.0
(isaaclab_venv) qiyuan@qiyuan-ue:~/Softwares/IsaacLab$ python -c "import isaaclab; print('Isaac Lab OK')"
Isaac Lab OK
(isaaclab_venv) qiyuan@qiyuan-ue:~/Softwares/IsaacLab$ python -c "from rsl_rl.runners import OnPolicyRunner; print('RSL_RL OK')"
RSL_RL OK



**2. 确认 Isaac Lab 2.3.2**

```bash
python -c "import isaaclab; print('Isaac Lab version:', isaaclab.__version__)"
```

(isaaclab_venv) qiyuan@qiyuan-ue:~/Softwares/IsaacLab$ python -c "import isaaclab; print('Isaac Lab version:', isaaclab.__version__)"
Isaac Lab version: 0.54.2



Isaac Lab 0.54.2 ✅（对应 git tag v2.3.2，内部版本号不同，是正常的）。

现在装 RSL_RL：

```bash
pip install rsl-rl-lib
```



**3. 确认 RSL_RL**

```bash
python -c "from rsl_rl.runners import OnPolicyRunner; print('RSL_RL OK')"
```

(isaaclab_venv) qiyuan@qiyuan-ue:~/Softwares/IsaacLab$ python -c "from rsl_rl.runners import OnPolicyRunner; print('RSL_RL OK')"
RSL_RL OK



**4. 运行 demo（headless）**

```bash
cd ~/Softwares/IsaacLab
python scripts/tutorials/00_sim/create_empty.py 
python scripts/tutorials/00_sim/create_empty.py --headless

python scripts/tutorials/00_sim/create_empty.py 2>&1 | tee ~/isaaclab_demo.log
```



系统 Python 3.11.15 ✅。重建 venv：

```bash
deactivate
rm -rf ~/Softwares/isaaclab_venv
/usr/bin/python3.11 -m venv ~/Softwares/isaaclab_venv
source ~/Softwares/isaaclab_venv/bin/activate

# 确认
which python
python --version
```

然后装 Isaac Sim 5.1.0：

```bash
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
```



正确！现在确认一下它不会引用 conda 的库：

```bash
python -c "import sys; print(sys.prefix); print(sys.base_prefix)"
```



两个都应该是 `/home/qiyuan/Softwares/isaaclab_venv`（不是 conda 路径）。

确认没问题就开始装：

```bash
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
```

完美！`base_prefix` 是 `/usr`（系统 Python），不是 conda 路径。这次对了。

开始装 Isaac Sim 5.1.0：

```bash
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
```



应该能。分析一下为什么这次会成功：

| 之前（conda）                                              | 现在（系统 venv）                                           |
| ---------------------------------------------------------- | ----------------------------------------------------------- |
| conda Python 3.11 → 自带 libstdc++ (GCC 13, CXXABI 1.3.15) | 系统 Python 3.11 → 用系统 libstdc++ (GCC 11, CXXABI 1.3.14) |
| Isaac Sim RTX 插件 → 系统 libstdc++                        | Isaac Sim RTX 插件 → 系统 libstdc++                         |
| **两者不一致 → 崩溃**                                      | **两者一致 → 兼容** ✅                                       |

之前崩溃的根因是 conda 的 libstdc++ 和 Isaac Sim 编译时用的 libstdc++ ABI 不兼容。现在系统 Python 和 Isaac Sim 都用同一个系统 libstdc++，不会有冲突。

装完后验证：

```bash
python scripts/tutorials/00_sim/create_empty.py --headless
```



**5. GPU 并行环境测试**

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Ant-v0 --headless --num_envs 32
```

按顺序跑，把结果贴出来。





五、最终环境

运行以下命令查看完整环境信息：

```bash
# Python 版本和路径
python --version && which python

# Isaac Lab 版本
python -c "import isaaclab; print('Isaac Lab:', isaaclab.__version__)"

# Isaac Sim（如果装了的话）
pip show isaacsim | grep Version

# PyTorch
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# RSL_RL
python -c "import rsl_rl; print('RSL_RL OK')" 2>/dev/null || echo "RSL_RL: 未安装"

# Newton
python -c "import isaaclab_newton; print('Newton OK')" 2>/dev/null || echo "Newton: 检查中..."
```



### 当前环境总结

| 组件          | 版本                           | 状态            |
| ------------- | ------------------------------ | --------------- |
| **操作系统**  | Ubuntu 22.04.5                 | ✅               |
| **GPU**       | NVIDIA RTX 4090 D (24GB)       | ✅               |
| **驱动**      | 595.84 / CUDA 13.2             | ✅               |
| **Python**    | 3.12.13（uv venv）             | ✅               |
| **PyTorch**   | 2.10.0+cu128                   | ✅               |
| **Isaac Lab** | 3.0.0-beta2（内部版本 6.1.17） | ✅               |
| **Isaac Sim** | 6.0.1.0                        | ✅（GUI 待验证） |
| **Newton**    | 已安装（kit-less 物理后端）    | ✅               |
| **RSL_RL**    | 已安装                         | ✅               |

### 环境路径

| 路径                                 | 说明                                   |
| ------------------------------------ | -------------------------------------- |
| `~/Softwares/IsaacLab/`              | Isaac Lab 源码（release/3.0.0-beta2）  |
| `~/Softwares/IsaacLab/env_isaaclab/` | Python 3.12 venv                       |
| `~/Softwares/isaaclab_venv/`         | 旧 venv（Python 3.11，已废弃，可删除） |

### 激活方式

```bash
source ~/Softwares/IsaacLab/env_isaaclab/bin/activate
```

### 两种运行模式

```bash
# Newton 模式（轻量，RL 训练用）
./isaaclab.sh train --rl_library rsl_rl --task=Isaac-Ant-v0 --num_envs=16 physics=newton_mjwarp

# Isaac Sim 模式（完整渲染）
python scripts/tutorials/00_sim/create_empty.py
```

### 待验证

- Isaac Sim 6.0.1 GUI 模式是否还会 RTX 崩溃
- RL 训练流程是否跑通

### 可清理

```bash
rm -rf ~/Softwares/isaaclab_venv   # 旧的 Python 3.11 venv
rm -rf ~/Softwares/IsaacLab_old     # 如果有的话
```



六、导入xml模型到isaacsim



七、训练和目标



```
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=16 --max_iterations=5 2>&1 | tee ~/xgb_train_test.log
```



这个命令在 **播放（play）一个训练好的舞蹈策略模型**，具体解析：

```bash
# 1. 激活 conda 环境
source /home/lushilin/miniconda3/etc/profile.d/conda.sh
conda activate env_roboparty

# 2. 进入项目目录
cd /home/lushilin/roboparty_ws/roboparty_train/robolab

# 3. 运行 RSL_RL 的播放脚本（推理/可视化）
HYDRA_FULL_ERROR=1 python scripts/rsl_rl/play_amp.py \
  --task RPO-Dance-Track-S3-V2-60S-Play \    # 任务名：舞蹈追踪，S3版本，60秒
  --device cuda:0 \                            # 用 GPU 0
  --num_envs 1 \                               # 只跑 1 个环境（可视化用）
  --checkpoint /home/lushilin/.../model_7998.pt \  # 加载第7998次迭代的模型权重
  --real-time                                  # 实时播放（不加速）
```

**总结**：这是在用 Isaac Lab + RSL_RL 框架，加载一个训练好的四足机器人舞蹈模型（`model_7998.pt`），在单个环境中实时播放机器人的舞蹈动作，用于可视化验证训练效果。

`play_amp.py` 中的 `amp` 可能指 **Adversarial Motion Priors**（对抗运动先验），是一种让机器人学习人类动作（如舞蹈）的 RL 方法。

和你现在的任务类似，只是你是训练运动策略（locomotion），他是播放舞蹈策略（dance）。



对！这就是 Isaac Sim 的 **可视化播放模式**。

从截图看：
- **Isaac Sim 5.1.0** 的完整 UI
- 一个 **人形机器人**（白色模型）在网格地面上跳舞
- 左上角有 **Simulation Settings** 面板（Fabric GPU 加速）
- 右侧有 **Stage** 层级和 **Property** 属性面板
- 右下角有 **Scene Debug Visualization** 调试选项

这就是 `play_amp.py --real-time` 的效果：加载训练好的模型权重，在 Isaac Sim 里实时渲染机器人动作。

**你的 xgb 四足狗也可以这样**：

1. **先训练**（已完成测试）：
```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=300
```

2. **再播放可视化**：
```bash
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint logs/rsl_rl/xgb_flat/2026-08-11_xxx/model_300.pt \
  --real-time
```

这样就会弹出 Isaac Sim UI，看到你的 xgb 四足狗在运动！

不过你现在用的是 Isaac Sim 6.0.1，界面可能略有不同，但功能一样。



八、训练可视化

可以的！有两种方式可视化：

## 方式 1：训练时带可视化（慢但直观）

去掉 `--headless`，训练时 Isaac Sim 会弹出窗口：

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=16 --max_iterations=300
```

> 注意：可视化时环境数要少（16~64），否则 GPU 渲染会卡。训练速度会变慢很多。

## 方式 2：训练完用 play 可视化（推荐）

先 headless 快速训练，再用 `play` 命令可视化效果：

```bash
# 1. 快速训练（无渲染）
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=300 --headless

# 2. 可视化播放（弹出 Isaac Sim UI）
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint logs/rsl_rl/xgb_flat/2026-08-11_xxx/model_300.pt
```

`play` 会弹出 Isaac Sim 窗口，你就能看到机器人在动。

## 方式 3：训练中途暂停查看

训练过程中按 **Ctrl+C** 会保存 checkpoint，然后用方式 2 的 play 命令查看当前效果。

---

**建议**：先用方式 1 跑 16 个环境、少量迭代（比如 50 次），快速确认机器人行为正常（能站起来、不飞出去），然后再用方式 2 大规模训练。

要不要先跑一个带可视化的短训练看看？

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=16 --max_iterations=50
```





找到了！IsaacLab 里有这些可视化 demo：

---

## 1. 四足机器人可视化 Demo（最相关）

**`scripts/demos/quadrupeds.py`** — 展示 7 种四足机器人（Anymal B/C/D、Unitree A1/Go1/Go2、Spot）

```bash
./isaaclab.sh -p scripts/demos/quadrupeds.py
```

会弹出 Isaac Sim UI，显示 7 个机器人在地面上随机抖动关节。

---

## 2. RL 训练后可视化（Play）

**`scripts/reinforcement_learning/rsl_rl/play.py`**（已弃用，用新命令）

```bash
# 新方式：可视化播放训练好的模型
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1
```

---

## 3. 基础教程

| 脚本                                      | 说明                   |
| ----------------------------------------- | ---------------------- |
| `tutorials/00_sim/create_empty.py`        | 空场景                 |
| `tutorials/00_sim/spawn_prims.py`         | 生成基本几何体         |
| `tutorials/01_assets/run_articulation.py` | 关节体（机器人）可视化 |

---

## 建议你现在跑这个

先跑四足 demo 确认 Isaac Sim UI 能正常弹出：

```bash
./isaaclab.sh -p scripts/demos/quadrupeds.py
```

如果这个能看到 UI，说明显示配置没问题，之后训练完用 `play` 命令也能看到你的 xgb 机器人在动。





九、接下来怎么做

现在我完全理解了 Matrix 的运控架构。让我帮你理清训练思路：

---

## Matrix 运控架构分析

### 核心发现

Matrix 的 `robot_mc` 是一个**多策略系统**，不是单一 RL 策略：

```
┌─────────────────────────────────────────────────────────┐
│                    robot_mc 控制架构                      │
├─────────────────────────────────────────────────────────┤
│  高层：FSM 状态机                                         │
│    PASSIVE → STANDUP → RL_MIX → ...                     │
│                                                          │
│  中层：20+ 个 ONNX 策略模型                               │
│    ├── mix_walk（默认行走）                               │
│    ├── balancestand（平衡站立）                           │
│    ├── backflip（后空翻）                                 │
│    ├── jump（跳跃）                                       │
│    ├── crawl（匍匐）                                      │
│    └── ...                                               │
│                                                          │
│  底层：PD 控制 + 重力补偿                                  │
│    Kp=20, Kd=0.7（RL 模式）                              │
│    力矩限幅 ±28 Nm                                        │
└─────────────────────────────────────────────────────────┘
```

### 关键参数（XG 平台）

| 参数       | 值                         | 说明                  |
| ---------- | -------------------------- | --------------------- |
| 控制频率   | 500Hz                      | `controller_dt=0.002` |
| 站立高度   | 0.32m                      | `body_height`         |
| 站立关节位 | ABAD=0, HIP=0.8, KNEE=-1.5 | 默认姿态              |
| RL PD 增益 | Kp=20, Kd=0.7              | 比训练计划里的保守    |
| 最大速度   | vx=3.0, vy=1.0, wz=3.0     | MPC 限制              |

---

## 你的训练计划应该怎么调整

### 当前问题

你现在的 Isaac Lab 配置是**通用四足 locomotion**（像 Unitree Go2），但 Matrix 的 xgb 有特定需求：

1. **站立高度 0.32m**（不是 0.65m）
2. **PD 增益 Kp=20, Kd=0.7**（不是 Kp=25, Kd=0.5）
3. **关节默认位 KNEE=-1.5**（不是 -0.8/-1.0）
4. **速度范围更大**（vx 最大 3.0 m/s）

### 建议：分阶段训练

**Phase 4a：先复现 Matrix 的 `balancestand`（平衡站立）**

这是最基础的能力，也是其他运动的前提：

```python
# 在 rough_env_cfg.py 中调整
self.scene.robot.init_state.pos = (0.0, 0.0, 0.32)  # 站立高度
self.scene.robot.actuators["base_legs"].stiffness = 20.0  # Kp
self.scene.robot.actuators["base_legs"].damping = 0.7     # Kd

# 默认关节位置（和 Matrix 一致）
joint_pos={
    ".*_ABAD_JOINT": 0.0,
    "(FAR|FBL)_HIP_JOINT": 0.8,
    "(RAR|RBL)_HIP_JOINT": 0.8,
    ".*_KNEE_JOINT": -1.5,
}
```

**Phase 4b：再训练 `mix_walk`（混合行走）**

在站立基础上，加入速度跟踪奖励。

---

## 下一步行动

要不要先修改配置，让训练环境和 Matrix 的参数对齐？我可以帮你：

1. 修改 `xgb.py` 的默认关节位置和 PD 增益
2. 修改 `flat_env_cfg.py` 的站立高度
3. 重新跑一次短训练验证

这样训练出来的策略才能直接部署到 Matrix 系统里。



修改完成！现在验证一下配置能否正常加载：


配置加载成功！现在跑一次短训练验证：


**修改完成！训练正常运行。**

## 修改总结

| 文件                | 修改内容                                                     | Matrix 参数                      |
| ------------------- | ------------------------------------------------------------ | -------------------------------- |
| `xgb.py`            | PD 增益 Kp=25→**20**, Kd=0.5→**0.7**                         | FSM_RL_Kp=20, Kd=0.7             |
| `xgb.py`            | 站立高度 0.65→**0.32m**                                      | body_height=0.32                 |
| `xgb.py`            | 默认关节位 HIP 前/后区分→**统一 0.8**, KNEE→**-1.5**         | hip_stand=0.8, knee_stand=-1.5   |
| `rough_env_cfg.py`  | 速度范围 vx∈[0,1]→**[0,3]**, vy∈[-0.3,0.3]→**[-1,1]**, wz∈[-1,1]→**[-3,3]** | cmpc_x_vel=3, y_vel=1, yaw_vel=3 |
| `rsl_rl_ppo_cfg.py` | 学习率 1e-3→**3e-4**                                         | 标准 locomotion RL               |
| `flat_env_cfg.py`   | 移除 Newton 物理配置                                         | 使用 Isaac Sim PhysX             |

## 下一步

现在可以开始正式训练了：

```bash
# 平坦地形训练 300 次迭代
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=300 --headless
```

或者先用少量环境测试可视化效果（需要显示环境）：

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=16 --max_iterations=100
```





十、现在在做什么？

## 详细解释：你现在在做什么

### 一、整体目标

你要用 **Isaac Lab + RSL_RL** 从零训练一个 RL 策略，替代 Matrix 系统中加密的 ONNX 模型。

```
原系统（robot_mc）                    目标系统
┌─────────────────────┐              ┌─────────────────────┐
│ 加密 ONNX 模型       │    替代为     │ 自训练 ONNX 模型     │
│ （无法修改/调试）     │   ────────→  │ （开源、可迭代）     │
─────────────────────┘              └─────────────────────┘
```

---

### 二、当前工作流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        训练流程                                      │
─────────────────────────────────────────────────────────────────────
│                                                                      │
│  1. 定义机器人配置 (xgb.py)                                          │
│     ↓                                                                │
│  2. 定义训练环境 (rough_env_cfg.py / flat_env_cfg.py)                │
│     ↓                                                                │
│  3. 定义 PPO 算法参数 (rsl_rl_ppo_cfg.py)                            │
│     ↓                                                                │
│  4. 注册 Gym 环境 (__init__.py)                                      │
│     ↓                                                                │
│  5. 运行训练脚本 (train.py)                                          │
│     ↓                                                                │
│  6. 输出：模型权重 (.pt) + TensorBoard 日志                          │
│                                                                      │
─────────────────────────────────────────────────────────────────────
```

---

### 三、涉及的文件及作用

#### 1. 机器人配置

**文件**：`~/Softwares/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/xgb.py`

**作用**：定义 xgb 机器人的物理属性

```python
XGB_ACTUATOR_CFG = DCMotorCfg(
    stiffness=20.0,   # PD 控制 Kp（Matrix: 20）
    damping=0.7,      # PD 控制 Kd（Matrix: 0.7）
    effort_limit=28.0 # 力矩限幅 ±28Nm
)

XGB_CFG = ArticulationCfg(
    spawn=MjcfFileCfg(
        asset_path="...xgb.xml"  # 加载 MJCF 模型
    ),
    init_state=InitialStateCfg(
        pos=(0, 0, 0.32),        # 站立高度 0.32m
        joint_pos={
            ".*_ABAD_JOINT": 0.0,    # 外展关节
            ".*_HIP_JOINT": 0.8,     # 大腿关节
            ".*_KNEE_JOINT": -1.5,   # 小腿关节
        }
    )
)
```

**类比**：这是机器人的"出生证明"——告诉仿真器它长什么样、多重、关节怎么动。

---

#### 2. 训练环境配置（粗糙地形）

**文件**：`~/Softwares/IsaacLab/source/isaaclab_tasks/.../config/xgb/rough_env_cfg.py`

**作用**：定义训练场景、奖励函数、观测空间

```python
class XgbRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # 机器人配置
        self.scene.robot = XGB_CFG
        
        # 速度命令范围（Matrix: vx=3, vy=1, wz=3）
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.0)
        
        # 奖励函数权重
        self.rewards.track_lin_vel_xy_exp.weight = 1.5  # 跟踪线速度
        self.rewards.flat_orientation_l2.weight = -2.5  # 惩罚倾斜
        
        # 终止条件
        self.terminations.base_contact.body_names = "base_link"  # 基座触地=失败
```

**类比**：这是"训练场"——定义机器人要学什么任务、怎么评分、什么时候算失败。

---

#### 3. 训练环境配置（平坦地形）

**文件**：`~/Softwares/IsaacLab/source/isaaclab_tasks/.../config/xgb/flat_env_cfg.py`

**作用**：继承粗糙地形配置，简化为平地训练

```python
class XgbFlatEnvCfg(XgbRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 改成平地
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        # 移除高度扫描（平地不需要）
        self.scene.height_scanner = None
```

**类比**：这是"初级训练场"——先在平地学会走，再去复杂地形。

---

#### 4. PPO 算法参数

**文件**：`~/Softwares/IsaacLab/source/isaaclab_tasks/.../config/xgb/agents/rsl_rl_ppo_cfg.py`

**作用**：定义 RL 算法的超参数

```python
class XgbFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    max_iterations = 300          # 训练 300 轮
    num_steps_per_env = 24        # 每轮每环境 24 步
    
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128]  # 策略网络结构
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=3.0e-4,     # 学习率
        gamma=0.99,               # 折扣因子
        clip_param=0.2,           # PPO 裁剪参数
    )
```

**类比**：这是"学习方法"——定义机器人怎么从经验中学习。

---

#### 5. Gym 环境注册

**文件**：`~/Softwares/IsaacLab/source/isaaclab_tasks/.../config/xgb/__init__.py`

**作用**：把配置注册成 Gym 环境，让训练脚本能找到

```python
gym.register(
    id="Isaac-Velocity-Flat-XGB-v0",
    entry_point="...",
    kwargs={"env_cfg_entry_point": "...XgbFlatEnvCfg"}
)
```

**类比**：这是"课程表"——告诉系统"Isaac-Velocity-Flat-XGB-v0"这个课用什么教材。

---

#### 6. 训练脚本

**文件**：`~/Softwares/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py`

**作用**：Isaac Lab 提供的训练入口，你不需要修改

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=300
```

**类比**：这是"教练"——负责执行训练流程。

---

### 四、训练时发生了什么

```
┌─────────────────────────────────────────────────────────────────┐
│  每次迭代（iteration）                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 收集数据：4096 个并行环境 × 24 步 = 98304 条经验             │
│     ┌──────────────────────────────────────────────────────┐    │
│     │  环境 1: 观测→动作→奖励→新观测                        │    │
│     │  环境 2: 观测→动作→奖励→新观测                        │    │
│     │  ...                                                  │    │
│     │  环境 4096: 观测→动作→奖励→新观测                     │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                  │
│  2. 更新策略：用 PPO 算法更新神经网络权重                        │
│     - 计算优势函数（哪些动作好）                                 │
│     - 更新策略网络（增加好动作的概率）                           │
│     - 更新价值网络（更准确估计未来奖励）                         │
│                                                                  │
│  3. 记录日志：reward、episode_length、各种指标                   │
│                                                                  │
│  4. 重复 300 次                                                  │
│                                                                  │
─────────────────────────────────────────────────────────────────┘
```

---

### 五、输出结果

训练完成后，你会得到：

```
~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/
└── 2026-08-11_xxx/
    ├── model_300.pt          # 训练好的模型权重
    ├── config.yaml           # 训练配置
    └── events.out.tfevents   # TensorBoard 日志
```

- **model_300.pt**：可以部署到 Matrix 的 ONNX 推理引擎
- **TensorBoard 日志**：可视化训练曲线（reward 是否上升）

---

### 六、你现在处于哪个阶段

```
Phase 1: 环境搭建          ✅ 完成
Phase 2: 模型导入          ✅ 完成（MJCF 加载成功）
Phase 3: 训练环境配置      ✅ 完成（刚对齐 Matrix 参数）
Phase 4: 奖励函数设计      ⏳ 当前（用默认奖励，后续可调）
Phase 5: 训练执行           下一步
```

**下一步**：开始正式训练，观察 reward 曲线，根据效果调整奖励函数。





根据官方文档，Isaac Lab 3.0 有 **4 种可视化方式**：

| 可视化器            | 特点                          | 适合场景          |
| ------------------- | ----------------------------- | ----------------- |
| **Kit** (Omniverse) | Isaac Sim 完整 UI，高保真渲染 | 你想看的"弹出 UI" |
| **Newton**          | 轻量 OpenGL，低开销           | 快速迭代调试      |
| **Rerun**           | 浏览器查看，可回放            | 远程/录制         |
| **Viser**           | 浏览器 3D 查看                | 分享/远程         |

---

## 你的情况：用 Isaac Sim 训练 + 可视化

### 方法 1：训练时直接可视化（Kit 可视化器）

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=16 --max_iterations=100 \
  --viz kit
```

这会弹出 **Isaac Sim UI**，你能看到机器人在动。

> 注意：`--headless` 已弃用，用 `--viz kit` 代替。

---

### 方法 2：轻量可视化（Newton 可视化器）

如果 Kit 太卡或显示有问题，用 Newton 可视化器（轻量 OpenGL）：

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=16 --max_iterations=100 \
  --viz newton
```

Newton 可视化器支持：
- **WASD/箭头键** 移动视角
- **鼠标拖拽** 环顾
- **滚轮** 缩放
- **H** 隐藏/显示 UI
- **ESC** 退出

---

### 方法 3：只可视化部分环境（推荐）

训练 4096 个环境时，只渲染其中几个：

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=300 \
  --viz kit --max_visible_envs 4
```

这样训练速度快（4096 并行），但只渲染 4 个环境给你看。

---

### 方法 4：训练完再可视化（Play 模式）

先 headless 训练，再用 `play` 命令可视化：

```bash
# 1. 训练（无渲染）
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=300

# 2. 可视化播放
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --viz kit
```

---

## 建议你现在试

先试 **Newton 可视化器**（轻量，不容易出显示问题）：

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=16 --max_iterations=50 \
  --viz newton
```

如果 Newton 可视化器能弹出窗口，再试 Kit 可视化器看 Isaac Sim 完整 UI。





运行这个命令后，你会得到 **两部分东西**：

---

## 一、训练过程中：实时可视化窗口

会弹出一个 **Newton 可视化窗口**（OpenGL 渲染），你能看到：

- 🤖 **16 个 xgb 四足机器人在地面上**
- 🎮 机器人会随机抖动、摔倒、尝试站起来（刚开始很笨）
- 📊 随着迭代进行，机器人逐渐学会保持平衡
- 🕹️ 你可以用 WASD 移动视角、鼠标拖拽环顾

**但注意**：Newton 可视化器比较简陋，不像 Isaac Sim 那样有完整的 UI 面板（Stage、Property 等）。它就是一个 3D 窗口显示仿真画面。

---

## 二、训练完成后：生成的文件

```
~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/
└── 2026-08-11_14-xx-xx/
    ├── model_50.pt              ← 训练好的模型权重（第50轮）
    ├── model_50.onnx            ← ONNX 格式（如果配置了导出）
    ├── config.yaml              ← 训练配置备份
    └── events.out.tfevents.xxx  ← TensorBoard 日志
```

### 各文件用途：

| 文件                    | 用途                                 |
| ----------------------- | ------------------------------------ |
| **model_50.pt**         | PyTorch 权重，可以加载继续训练或部署 |
| **config.yaml**         | 记录训练时的所有参数，方便复现       |
| **events.out.tfevents** | TensorBoard 日志，可视化训练曲线     |

---

## 三、终端输出

训练过程中终端会打印每轮迭代的信息：

```
Learning iteration 10/50
    Mean reward: -0.85          ← 奖励（越高越好，目标是接近 0 或正数）
    Mean episode length: 45.2   ← 平均存活步数（越长说明越稳定）
    success_rate: 0.35          ← 速度跟踪成功率
    base_contact: 0.15          ← 摔倒率（越低越好）
```

---

## 四、50 次迭代够吗？

**不够。** 50 次只是验证配置能不能跑通。正式训练需要：

| 阶段     | 迭代次数  | 预期效果                   |
| -------- | --------- | -------------------------- |
| 验证配置 | 5~50      | 能跑通，机器人可能还站不稳 |
| 初步收敛 | 300~500   | 能站立、缓慢行走           |
| 正式训练 | 1000~3000 | 稳定行走、速度跟踪         |

---

## 建议

先跑 50 次看可视化效果，确认机器人行为正常（不飞出去、不穿透地面），然后加大规模训练：

```bash
# 正式训练
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=1000 \
  --viz newton --max_visible_envs 4
```

这样你能一边看 4 个机器人的实时表现，一边用 4096 个环境快速训练。





**xgb_flat 目录是空的**，之前的训练输出 `2026-08-11_15-48-27` 已经不在了（可能被清理了）。

你需要**重新训练**才能生成模型文件：

```bash
# 训练 50 次迭代（验证用）
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=16 --max_iterations=50
```

训练完成后，用 `play` 命令导出 ONNX：

```bash
# 先找到生成的模型路径
ls ~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/

# 然后用 play 导出（替换成实际的时间戳目录）
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint /home/qiyuan/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/2026-08-11_16-41-28/model_49.pt
```

`play` 运行后会自动在模型同目录下生成 `exported/policy.onnx`。





进程已经结束了。让我看看有没有生成 ONNX 文件：


**ONNX 导出成功！** 🎉

```
~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/2026-08-11_16-41-28/
├── model_49.pt              ← 训练好的 PyTorch 模型
├── exported/
│   ├── policy.pt            ← JIT 格式
│   ├── policy.onnx          ← ONNX 模型 ✅
│   └── policy.onnx.data     ← ONNX 权重数据
└── params/
    ├── agent.yaml           ← PPO 配置
    └── env.yaml             ← 环境配置
```

---

## 检查 ONNX 模型结构

```bash
# 用 Python 查看 ONNX 输入输出
cd ~/Softwares/IsaacLab
./isaaclab.sh -p -c "
import onnx
model = onnx.load('logs/rsl_rl/xgb_flat/2026-08-11_16-41-28/exported/policy.onnx')
print('输入:', [(i.name, list(i.type.tensor_type.shape.dim)) for i in model.graph.input])
print('输出:', [(o.name, list(o.type.tensor_type.shape.dim)) for o in model.graph.output])
"
```

这会告诉你 ONNX 模型的输入输出张量名和维度，方便后续集成到 Matrix。





**结果完全一样！** ✅

| 对比项      | 17-00-11      | 17-02-19      | 结论         |
| ----------- | ------------- | ------------- | ------------ |
| 环境配置    | ✅ 相同        | ✅ 相同        | 一致         |
| 算法配置    | ✅ 相同        | ✅ 相同        | 一致         |
| model_50.pt | `1ccde98b...` | `1ccde98b...` | **完全相同** |
| model_99.pt | 983841 bytes  | 983841 bytes  | 大小相同     |

---

## 为什么一样？

**可视化器不影响训练结果**。可视化器只是：
- **Kit**：用 Isaac Sim 渲染（高保真，慢）
- **Newton**：用 OpenGL 渲染（轻量，快）

它们都使用**相同的物理引擎（PhysX）**和**相同的 RL 算法（PPO）**，只是渲染方式不同。

---

## 可视化器的作用

| 可视化器              | 训练速度 | 渲染质量 | 适合场景      |
| --------------------- | -------- | -------- | ------------- |
| **无** (`--headless`) | 最快     | 无       | 大规模训练    |
| **Newton**            | 快       | 中等     | 调试/快速查看 |
| **Kit**               | 慢       | 高       | 最终演示/截图 |

---

## 建议

- **训练时**：用 `--headless` 或 `--viz newton`（快）
- **演示时**：用 `--viz kit`（好看）

你现在可以用任何一个目录里的模型，比如：

```bash
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint /home/qiyuan/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/2026-08-11_17-02-19/model_99.pt \
  --viz kit
```

这样能在 Isaac Sim UI 里看到机器人行走了。





**完全可以！** 这是强化学习的标准做法。

---

## 训练 vs 部署的区别

| 阶段     | 环境数量  | 目的                         |
| -------- | --------- | ---------------------------- |
| **训练** | 16 个并行 | 加速学习，收集更多经验       |
| **部署** | 1 个      | 用训练好的策略控制真实机器人 |

---

## 为什么训练时用 16 个？

```
┌─────────────────────────────────────────────────────────────────┐
│  训练阶段（加速学习）                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  环境 1: 机器人 A 尝试走 → 摔倒 → 学到教训                       │
│  环境 2: 机器人 B 尝试走 → 成功 → 学到经验                       │
│  环境 3: 机器人 C 尝试走 → 摔倒 → 学到教训                       │
│  ...                                                             │
│  环境 16: 机器人 P 尝试走 → 成功 → 学到经验                      │
│                                                                  │
│  16 个环境并行 = 16 倍经验收集速度                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 部署时怎么用？

训练好的 `policy.onnx` 就是一个**神经网络函数**：

```python
# 输入：48 维观测（当前状态）
obs = [
    base_lin_vel,      # 3  基座线速度
    base_ang_vel,      # 3  基座角速度
    projected_gravity, # 3  重力投影
    velocity_cmd,      # 3  速度命令
    joint_pos,         # 12 关节位置
    joint_vel,         # 12 关节速度
    last_actions,      # 12 上一步动作
]  # 总共 48 维

# 输出：12 维关节目标位置
actions = policy(obs)  # [12] 关节目标位置偏移
```

**这个函数是通用的**，不管你有 1 个还是 100 个机器人，只要：
1. 构造相同的 48 维观测
2. 调用 `policy.onnx` 推理
3. 得到 12 维动作

---

## Matrix 中的部署

```
┌─────────────────────────────────────────────────────────────────┐
│  Matrix 部署（单机器人）                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  传感器数据 → 构造 48 维观测 → policy.onnx → 12 维动作           │
│                                                                  │
│  控制循环（500Hz）：                                             │
│  1. 读取 IMU、关节编码器                                         │
│  2. 构造 obs 张量                                                │
│  3. 调用 ONNX 推理                                               │
│  4. 输出关节目标位置给 PD 控制器                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 关键问题：观测怎么构造？

Matrix 部署时需要构造**相同的 48 维观测**：

| 观测项              | 维度 | 来源                  |
| ------------------- | ---- | --------------------- |
| `base_lin_vel`      | 3    | IMU 或 odom 估计      |
| `base_ang_vel`      | 3    | IMU 陀螺仪            |
| `projected_gravity` | 3    | IMU 加速度计          |
| `velocity_cmd`      | 3    | 用户命令（手柄/键盘） |
| `joint_pos`         | 12   | 关节编码器            |
| `joint_vel`         | 12   | 关节编码器差分        |
| `last_actions`      | 12   | 上一步策略输出        |

---

## 总结

| 问题                          | 答案                     |
| ----------------------------- | ------------------------ |
| 训练 16 个，部署 1 个能用吗？ | ✅ 完全可以               |
| 需要修改 ONNX 吗？            | ❌ 不需要                 |
| 需要修改观测构造吗？          | ✅ 需要确保 48 维观测一致 |
| 需要修改动作输出吗？          | ✅ 需要确保 12 维动作一致 |

**下一步**：在 MuJoCo 里测试 `policy.onnx`，确认单机器人能正常行走，再集成到 Matrix。



## 为什么用 model_99.pt 或 model_49.pt？

因为你的训练命令是：

```bash
--max_iterations=100
```

训练会保存多个检查点：

| 文件          | 含义                                |
| ------------- | ----------------------------------- |
| `model_0.pt`  | 初始模型（随机权重，还没训练）      |
| `model_49.pt` | 第 50 轮模型（训练了一半）          |
| `model_99.pt` | 第 100 轮模型（训练完成，最终模型） |

**用最新的（model_99.pt）**：因为训练越久，模型学得越好。

---

## 为什么用最新的目录？

每次训练都会生成一个带时间戳的新目录：

```
logs/rsl_rl/xgb_flat/
├── 2026-08-11_15-48-27/   ← 之前的训练（已删除）
├── 2026-08-11_17-00-11/   ← 某次训练
└── 2026-08-11_17-02-19/   ← 最新训练（你刚才跑的）
```

**用最新的目录**：因为你刚才跑的训练结果在那里。

---

## 如果不确定用哪个

```bash
# 1. 查看所有训练目录
ls ~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/

# 2. 查看某个目录里有哪些模型
ls ~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/2026-08-11_17-02-19/

# 3. 用最新目录里编号最大的模型
./isaaclab.sh play ... --checkpoint .../model_99.pt
```

**简单说**：编号越大 = 训练越久 = 模型越好。





命令**没有问题**，它在正常运行。

---

## 为什么没有结束？

`play` 命令会**一直运行仿真循环**，直到：
- 达到 `max_steps`（默认可能是 1000 步或无限）
- 你按 **Ctrl+C** 终止

---

## 它在干什么？

从输出可以看到：

```
[INFO]: Loading model checkpoint from: .../model_99.pt
...
Actor Model: MLPModel(...)
Critic Model: MLPModel(...)
```

模型已加载，现在在**实时仿真**：
- 1 个环境
- 用训练好的策略控制机器人
- 每步输出 12 维关节动作

---

## 如何终止？

按 **Ctrl+C** 结束 play 命令。

---

## 如何导出 ONNX？

ONNX 会在 play 运行时**自动生成**在：

```
logs/rsl_rl/xgb_flat/2026-08-11_17-02-19/exported/
├── policy.pt
└── policy.onnx
```

你可以**现在按 Ctrl+C 终止**，然后检查：

```bash
ls ~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/2026-08-11_17-02-19/exported/
```

如果 ONNX 已经生成，就可以用了。





这是**正常现象**——100 次迭代远远不够，策略还没学会走路，只会随机抖动关节。

---

## 为什么会"跳舞"？

| 原因                | 说明                                             |
| ------------------- | ------------------------------------------------ |
| **训练不足**        | 100 次迭代 ≈ 刚学会"不要马上摔倒"，还没学会走路  |
| **策略输出随机**    | 神经网络权重还没收敛，输出的是接近随机的关节偏移 |
| **PD 控制放大噪声** | Kp=20 会把小的随机偏移放大成剧烈动作             |

---

## 解决方案：训练更多迭代

```bash
cd ~/Softwares/IsaacLab

# 正式训练 1000 次迭代（推荐）
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=1000
```

训练时间参考：
- 100 次迭代 ≈ 20 秒
- 1000 次迭代 ≈ 3~5 分钟
- 3000 次迭代 ≈ 10~15 分钟

---

## 训练完成后重新导出 ONNX

```bash
# 找到最新模型
ls ~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/

# 导出 ONNX（替换成实际目录）
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint /home/qiyuan/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/<最新目录>/model_999.pt
```

---

## 预期效果对比

| 迭代次数 | 行为                     |
| -------- | ------------------------ |
| 100      | 🕺 疯狂跳舞（当前）       |
| 300~500  | 🧍 能站立，偶尔走几步     |
| 1000     | 🚶 能稳定行走，但不够流畅 |
| 3000     | 🏃 流畅行走，速度跟踪良好 |

---

## 快速验证：先试 500 次

```bash
cd ~/Softwares/IsaacLab

./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=500
```

500 次迭代后重新导出 ONNX，再跑 MuJoCo 脚本，你应该能看到机器人**站立并尝试行走**，而不是跳舞了。



又崩溃了，而且更严重。看关键指标：

```
Mean value loss: nan          ← 价值网络发散了！
Mean reward: -2038122.13      ← 奖励爆炸
action_rate_l2: -53078.44     ← 动作剧烈跳动
```

**根本原因**：价值网络（critic）先发散 → 优势函数计算错误 → 策略网络崩溃。

`std_type="log"` 没解决问题，因为问题不在 std，而在**学习率太高导致数值不稳定**。

---

## 修复：降低学习率 + 增强稳定性


有重复参数，让我修复：


配置已修复，没有语法错误了。

---

## 本次修改汇总

| 参数                  | 修改前 | 修改后   | 原因                            |
| --------------------- | ------ | -------- | ------------------------------- |
| `init_std`            | 1.0    | **0.5**  | 初始探索范围太大，动作爆炸      |
| `std_type`            | scalar | **log**  | 保证 std 始终为正               |
| `learning_rate`       | 3e-4   | **1e-4** | 学习率太高，价值网络发散（NaN） |
| `entropy_coef`        | 0.01   | **0.02** | 增加探索稳定性                  |
| `num_learning_epochs` | 5      | **3**    | 减少每轮过拟合                  |

---

## 崩溃原因分析

```
Mean value loss: nan          ← 价值网络先发散
Mean reward: -2038122.13      ← 奖励爆炸
action_rate_l2: -53078.44     ← 动作剧烈跳动
Mean action std: 1.20         ← std 已经很大了
```

**连锁反应**：
1. 学习率 3e-4 太高 → 价值网络权重更新过大 → NaN
2. 价值网络 NaN → 优势函数计算错误 → 策略梯度方向错误
3. 策略输出极端动作 → `action_rate_l2` 惩罚爆炸 → 奖励 -200 万

---

## 重新训练

```bash
cd ~/Softwares/IsaacLab

./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=3000
```

这次应该能稳定跑到 3000 次迭代。如果还崩溃，可能需要进一步降低学习率到 `5e-5`。





根据 `robot_mc_analysis.md` 的分析，Matrix 的运控系统由 **FSM 状态机 + RL 策略** 组成。你描述的功能拆解如下：

---

## Matrix 的 FSM 状态机

```
PASSIVE ──(U键)──→ STANDUP ──(完成)──→ RL_MIX ←──(WASDQE 控制)──┐
   ↑                      │                                      │
   │                      │                                      │
   └────(Space键)─────────┴──────────────────────────────────────┘
```

| 状态        | 控制方式                 | 说明                     |
| ----------- | ------------------------ | ------------------------ |
| **PASSIVE** | 纯阻尼（Kd=3.0）         | 趴在地上，电机自由       |
| **STANDUP** | PD 插值（Kp=80）         | 从趴下→站立，**不是 RL** |
| **RL_MIX**  | RL 策略（Kp=20, Kd=0.7） | 各种运动模式             |

---

## 你需要训练的模型

### 1. 站立/趴下：**不需要训练 RL 模型** ✅

这是 **传统 PD 控制**，直接插值关节角度：

```
趴下姿态: hip=1.4, knee=-2.4
    ↓ PD 插值 (Kp=80, Kd=1.0, 3秒)
站立姿态: hip=0.8, knee=-1.5
```

**实现方式**：在 MuJoCo 脚本中用代码实现，不需要神经网络。

---

### 2. 行走控制（WASDQE）：**需要 1 个 RL 模型**

对应 Matrix 的 **`mix_walk`** 策略：

| 按键 | 命令 | 速度范围             |
| ---- | ---- | -------------------- |
| W    | 前进 | vx ∈ [0, 3.0] m/s    |
| S    | 后退 | vx ∈ [-1.0, 0] m/s   |
| A    | 左移 | vy ∈ [0, 1.0] m/s    |
| D    | 右移 | vy ∈ [-1.0, 0] m/s   |
| Q    | 左转 | wz ∈ [0, 3.0] rad/s  |
| E    | 右转 | wz ∈ [-3.0, 0] rad/s |

**这就是你已经在训练的模型！** `Isaac-Velocity-Flat-XGB-v0`

---

### 3. 平衡站立：**可选，需要 1 个 RL 模型**

对应 Matrix 的 **`balancestand`** 策略：
- 原地保持平衡
- 抵抗外力扰动
- 支持偏航旋转

---

## 总结：你需要训练几个模型？

| 功能           | 模型类型                | 数量  | 状态         |
| -------------- | ----------------------- | ----- | ------------ |
| 趴下 ↔ 站立    | PD 插值（代码实现）     | 0     | ✅ 不需要训练 |
| 行走（WASDQE） | RL 策略（mix_walk）     | **1** | 正在训练     |
| 平衡站立       | RL 策略（balancestand） | 1     | ⏳ 可选       |

**核心答案：你只需要训练好当前的 `mix_walk` 模型就够了。**

---

## 当前训练的问题

你的训练在第 1389 次崩溃了，我已经修复了配置（降低学习率、减小 init_std）。重新训练：

```bash
cd ~/Softwares/IsaacLab

./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-XGB-v0 \
  --num_envs=4096 --max_iterations=3000
```

训练完成后导出 ONNX，集成到 MuJoCo 脚本中，就能实现：
- **U 键**：PD 插值站立（代码实现）
- **Space 键**：PD 插值趴下（代码实现）
- **WASDQE**：ONNX 策略推理（行走控制）



让我查看 Matrix 文档中关于这两个策略的详细说明：


让我搜索更多关于这两个策略的区别：


根据文档和命名惯例，这两个策略的区别是：

---

## `mix_walk` vs `gait_walk` 对比

| 特性         | `mix_walk`（混合行走）   | `gait_walk`（步态行走） |
| ------------ | ------------------------ | ----------------------- |
| **说明**     | 混合行走（默认 RL 步态） | 步态行走                |
| **步态生成** | RL 自适应生成            | 预定义步态模式          |
| **灵活性**   | 高（适应复杂地形）       | 中（适合平坦地面）      |
| **稳定性**   | 依赖 RL 训练质量         | 更稳定可预测            |
| **用途**     | 通用行走、越障           | 简单直线行走            |
| **复杂度**   | 高（需要更多训练）       | 低（训练更快收敛）      |

---

## 命名规律

从 Matrix 的其他策略可以看出：

**`mix_` 前缀** = RL 混合控制（更复杂、自适应）
- `mix_walk` - 混合行走
- `mix_backflip` - 混合后空翻
- `mix_flipover` - 混合翻身
- `mix_climb` - 混合攀爬
- `mix_crawl` - 混合匍匐

**`gait_` 前缀** = 传统步态控制（更简单、稳定）
- `gait_walk` - 步态行走
- `measured` - 测量步态（可能是从真实机器人采集的）

---

## 你的情况

你当前训练的 `xgb_flat` 模型**相当于 `mix_walk`**，因为：
1. 使用 Isaac Lab 的 RL 训练
2. 观测空间包含速度命令、关节状态等
3. 策略自适应生成步态

如果你想训练 `gait_walk` 风格的模型，需要：
- 使用更结构化的奖励函数（强制特定步态模式）
- 或者使用参考动作跟踪（Reference State Initialization）

---

## 建议

**保持当前训练**，继续到 8000 次迭代。如果直线行走问题解决了，你的模型就相当于 `mix_walk`，是 Matrix 的默认行走策略。





**是的，你的模型相当于 `mix_walk`。**

---

## 对比分析

| 特性         | 你的 `xgb_flat` 模型                 | Matrix `mix_walk`                    |
| ------------ | ------------------------------------ | ------------------------------------ |
| **训练框架** | Isaac Lab + RSL_RL PPO               | 自研框架 + PPO                       |
| **观测空间** | 48 维（速度、关节、命令等）          | 类似（推测 48-60 维）                |
| **动作空间** | 12 关节目标位置                      | 12 关节目标位置 + tau_ff（前馈力矩） |
| **速度命令** | vx ∈ [0,1], vy ∈ [-1,1], wz ∈ [-3,3] | 类似范围                             |
| **地形**     | 平地（flat）                         | 平地 + 随机地形                      |
| **步态生成** | RL 自适应                            | RL 自适应                            |
| **用途**     | 通用行走                             | 通用行走（默认模式）                 |

---

## 关键差异

### 1. **动作空间**
你的模型只输出 **关节目标位置**，Matrix 的 `mix_walk` 还输出 **前馈力矩 `tau_ff`**：
```python
# 你的模型
actions = policy(obs)  # [12] 关节目标位置

# Matrix mix_walk
actions, tau_ff = policy(obs)  # [12] + [12] 关节位置 + 前馈力矩
```

这就是为什么你的 MuJoCo 脚本需要手动添加重力补偿，而 Matrix 不需要（`tau_ff` 已经包含了）。

### 2. **训练复杂度**
Matrix 的 `mix_walk` 可能在更复杂的环境下训练：
- 随机地形（台阶、斜坡、凹凸）
- 域随机化（质量、摩擦、延迟）
- 课程学习（从简单到困难）

你的模型只在**平地**上训练，所以可能不够鲁棒。

### 3. **观测空间**
Matrix 可能还有额外的观测：
- 步态相位（gait phase）
- 接触传感器（foot contact）
- 历史动作序列（action history）

---

## 如何更接近 `mix_walk`

如果你想让模型更接近 Matrix 的 `mix_walk`，可以：

### 方案 1：在复杂地形上训练（推荐）
修改 `rough_env_cfg.py`，启用粗糙地形：
```python
# 使用 rough 环境而不是 flat
self.terrain.mesh_type = "trimesh"  # 随机地形
```

### 方案 2：增加前馈力矩输出
修改网络结构，输出 24 维动作（12 位置 + 12 力矩）：
```python
# 需要修改网络配置
actor_hidden_dims = [256, 128, 64]
# 输出层改为 24 维
```

### 方案 3：增加更多观测
添加步态相位、接触传感器等观测。

---

## 总结

**你的模型 ≈ `mix_walk` 的简化版**

- ✅ 核心思想相同（RL 自适应步态）
- ⚠️ 缺少前馈力矩输出
- ⚠️ 训练环境较简单（只有平地）

当前模型如果训练到 8000 次后能稳定直线行走，就可以作为你的 `mix_walk` 使用。如果需要更强的鲁棒性，再考虑增加地形复杂度和观测维度。





让我查看 Isaac Lab 官方支持的机器人：


**有的！Isaac Lab 官方支持多款四足机器人：**

---

## 官方四足机器人列表

| 机器人              | 文件         | 说明                         |
| ------------------- | ------------ | ---------------------------- |
| **ANYmal**          | `anymal.py`  | ANYbotics 四足（最常用示例） |
| **Spot**            | `spot.py`    | Boston Dynamics Spot         |
| **Unitree Go1/Go2** | `unitree.py` | 宇树 Go1、Go2                |
| **Unitree B1/B2**   | `unitree.py` | 宇树 B1、B2                  |
| **Unitree Aliengo** | `unitree.py` | 宇树 Aliengo                 |
| **Ant**             | `ant.py`     | 简单蚂蚁机器人（教学用）     |
| **XGB**             | `xgb.py`     | 你的自定义机械狗 ✅           |

---

## ANYmal 示例（最经典）

ANYmal 是 Isaac Lab 的**官方示例机器人**，有完整的 locomotion 任务：

```bash
# 查看 ANYmal 的行走任务
ls /home/qiyuan/Softwares/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/

# 应该能看到 anymal 相关配置
```

---

## 你的 XGB 与官方示例对比

| 特性         | ANYmal（官方）                  | XGB（你的）                  |
| ------------ | ------------------------------- | ---------------------------- |
| **关节数**   | 12                              | 12                           |
| **观测空间** | 48 维                           | 48 维                        |
| **动作空间** | 12 维                           | 12 维                        |
| **训练任务** | `Isaac-Velocity-Flat-Anymal-v0` | `Isaac-Velocity-Flat-XGB-v0` |
| **收敛速度** | ~1500 次                        | ~3000 次                     |

---

## 参考 ANYmal 的训练配置

如果你想参考官方示例，可以查看：


**是的！Isaac Lab 有完整的官方四足机器人示例：**

---

## 官方 ANYmal 配置

```
/home/qiyuan/Softwares/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/
├── anymal_b/          # ANYmal B 版本
├── anymal_c/          # ANYmal C 版本（最常用）
├── anymal_d/          # ANYmal D 版本
└── xgb/               # 你的 XGB ✅
```

---

## 你可以参考 ANYmal 的配置

查看 ANYmal 的训练参数：

```bash
# 查看 ANYmal 的奖励函数配置
cat /home/qiyuan/Softwares/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/rough_env_cfg.py

# 查看 ANYmal 的 PPO 配置
cat /home/qiyuan/Softwares/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_c/agents/rsl_rl_ppo_cfg.py
```

---

## 运行 ANYmal 官方示例

你可以直接运行 ANYmal 的预训练模型，看看官方效果：

```bash
cd ~/Softwares/IsaacLab

# Play ANYmal 的行走策略
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-Anymal-C-Play-v0 \
  --num_envs=1 \
  --viz kit
```

或者训练 ANYmal：

```bash
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-Anymal-C-v0 \
  --num_envs=4096 \
  --max_iterations=1500
```

---

## 对比学习

通过对比 ANYmal 和你的 XGB，你可以：
1. **学习奖励函数设计**（官方如何惩罚侧向移动、倾斜等）
2. **学习域随机化参数**（官方如何增加鲁棒性）
3. **学习课程学习策略**（官方如何从简单到困难）

这样你就能理解为什么 ANYmal 在 1500 次就能收敛，而你的 XGB 需要 3000+ 次。





错误是因为**没有 ANYmal 的预训练模型**。你需要先训练或者下载预训练模型。

---

## 解决方案

### 方案 1：先训练 ANYmal（推荐）

```bash
cd ~/Softwares/IsaacLab

# 训练 ANYmal（1500 次迭代）
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-Anymal-C-v0 \
  --num_envs=4096 \
  --max_iterations=1500
```

训练完成后，模型会保存在：
```
/home/qiyuan/Softwares/IsaacLab/logs/rsl_rl/anymal_c_flat/
```

然后再运行 play：

```bash
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-Anymal-C-Play-v0 \
  --num_envs=1 \
  --viz kit
```

---

### 方案 2：查看你已有的模型

让我看看你的日志目录里有什么：


你只有 `xgb_flat` 的模型，没有 ANYmal 的预训练模型。

---

## 立即行动：训练 ANYmal

```bash
cd ~/Softwares/IsaacLab

# 训练 ANYmal（约 15-30 分钟）
./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-Anymal-C-v0 \
  --num_envs=4096 \
  --max_iterations=1500
```

训练完成后，你就可以：

1. **Play 查看效果**：
```bash
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-Anymal-C-Play-v0 \
  --num_envs=1 \
  --viz kit
```

2. **对比你的 XGB**：
   - ANYmal 1500 次 vs XGB 3000 次
   - 看哪个走得更稳、更直

---

## 预期效果

ANYmal 官方示例在 1500 次迭代后应该能：
- ✅ 稳定直线行走
- ✅ 身体保持水平
- ✅ 速度跟踪准确
- ✅ 连续走 30 秒+ 不摔倒

这样你就能直观看到**训练良好的模型应该是什么样的**。







好的！让我帮你设置 Unitree Go2 的训练。先查看 Isaac Lab 中 Go2 的配置：


太好了！**Isaac Lab 已经内置了 Unitree Go2 的完整配置**，可以直接训练！

---

## Unitree Go2 可用任务

| 任务名                                     | 说明          |
| ------------------------------------------ | ------------- |
| `Isaac-Velocity-Flat-Unitree-Go2-v0`       | 平地训练      |
| `Isaac-Velocity-Flat-Unitree-Go2-Play-v0`  | 平地 Play     |
| `Isaac-Velocity-Rough-Unitree-Go2-v0`      | 粗糙地形训练  |
| `Isaac-Velocity-Rough-Unitree-Go2-Play-v0` | 粗糙地形 Play |

---

## 立即开始训练

### 1. 平地训练（推荐先从这个开始）

```bash
cd ~/Softwares/IsaacLab

./isaaclab.sh train --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs=4096 \
  --max_iterations=1500
```

### 2. 训练完成后 Play 查看效果

```bash
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --num_envs=1 \
  --viz kit
```

### 3. 导出 ONNX（可选）

训练完成后，导出模型用于 MuJoCo 部署：

```bash
./isaaclab.sh play --rl_library rsl_rl \
  --task=Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --num_envs=1 \
  --checkpoint /home/qiyuan/Softwares/IsaacLab/logs/rsl_rl/go2_flat/<timestamp>/model_1499.pt
```

---

## 预期效果

Go2 官方配置应该在 **1500 次迭代**后达到：
- ✅ 稳定直线行走
- ✅ 身体保持水平
- ✅ 速度跟踪准确
- ✅ 连续行走不摔倒

---

## 对比你的 XGB

| 机器人  | 迭代次数 | 预期效果                 |
| ------- | -------- | ------------------------ |
| **Go2** | 1500     | 稳定行走                 |
| **XGB** | 3000     | 基本行走（还有拐弯问题） |

训练 Go2 后，你可以对比两个模型的效果，看看官方配置为什么收敛更快。

---

要开始训练吗？





## 影响四足机器人 RL 训练质量的核心因素

---

### 1. **迭代次数**（最直接）

| 迭代次数   | 效果         |
| ---------- | ------------ |
| < 1000     | 基本学不会   |
| 1000-3000  | 能走但不稳定 |
| 3000-5000  | 稳定行走     |
| 5000-10000 | 鲁棒行走     |
| 10000+     | 高级技能     |

你的情况：
- XGB：3000 次 → 能走但拐弯
- Go2：1500 次 → 能走但拐弯

**都还不够。**

---

### 2. **奖励函数设计**（最关键）

奖励函数决定策略"学什么"。核心奖励项：

| 奖励项                 | 作用               | 权重建议 |
| ---------------------- | ------------------ | -------- |
| `track_lin_vel_xy_exp` | 跟踪速度命令       | +1.5     |
| `track_ang_vel_z_exp`  | 跟踪转向命令       | +0.75    |
| `lin_vel_z_l2`         | 惩罚 Z 方向速度    | -2.0     |
| `flat_orientation_l2`  | 惩罚身体倾斜       | -2.5     |
| `dof_torques_l2`       | 惩罚大力矩（节能） | -0.0002  |
| `action_rate_l2`       | 惩罚动作突变       | -0.01    |
| `feet_air_time`        | 鼓励有节奏步态     | +0.25    |

**缺少 `lin_vel_y_l2` 和 `ang_vel_z_l2` 惩罚 → 策略不关心侧向移动和转向 → 拐弯**

---

### 3. **速度命令范围**（影响收敛速度）

| 配置                  | 效果                         |
| --------------------- | ---------------------------- |
| `lin_vel_x ∈ [0, 1]`  | 只前进，任务简单，收敛快     |
| `lin_vel_x ∈ [-1, 1]` | 前后都能走，任务复杂，收敛慢 |
| `ang_vel_z ∈ [-1, 1]` | 小范围转向，收敛快           |
| `ang_vel_z ∈ [-3, 3]` | 大范围转向，收敛慢           |

**命令范围越大 → 策略要学的东西越多 → 需要更多迭代**

---

### 4. **域随机化**（影响鲁棒性）

| 随机化项   | 作用         |
| ---------- | ------------ |
| 质量随机化 | 适应不同负载 |
| 摩擦随机化 | 适应不同地面 |
| 推力随机化 | 抗干扰能力   |
| 延迟随机化 | 适应控制延迟 |
| 观测噪声   | 抗传感器噪声 |

**随机化太少 → 过拟合训练环境 → 部署时表现差**

---

### 5. **网络结构**（影响表达能力）

| 参数                 | 默认值         | 说明       |
| -------------------- | -------------- | ---------- |
| `actor_hidden_dims`  | [256, 128, 64] | 策略网络层 |
| `critic_hidden_dims` | [256, 128, 64] | 价值网络层 |
| `learning_rate`      | 0.001          | 学习率     |
| `num_envs`           | 4096           | 并行环境数 |

**网络太小 → 学不会复杂策略；网络太大 → 训练慢**

---

### 6. **物理参数**（影响真实性）

| 参数     | 影响                         |
| -------- | ---------------------------- |
| 仿真步长 | 太小 → 训练慢；太大 → 不稳定 |
| 控制频率 | 50Hz 是标准                  |
| 关节限幅 | 防止超限                     |
| 力矩限幅 | 防止过大输出                 |

---

## 你的模型问题诊断

| 问题     | 根因                | 解决方案                      |
| -------- | ------------------- | ----------------------------- |
| XGB 拐弯 | 迭代不够 + 缺少惩罚 | 继续训练到 8000 次 + 添加惩罚 |
| Go2 拐弯 | 迭代不够 + 缺少惩罚 | 继续训练到 3000 次 + 添加惩罚 |
| Go2 倒退 | heading 范围太大    | Play 时固定 heading=0         |

---

## 优先级排序

**快速改善训练质量的顺序：**

1. **增加迭代次数**（最简单，效果最明显）
2. **添加惩罚项**（`lin_vel_y_l2`、`ang_vel_z_l2`）
3. **缩小速度命令范围**（减少学习任务）
4. **调整奖励权重**（微调）
5. **增加域随机化**（提高鲁棒性）

---

## 建议行动

**先做第 1 步**：继续训练到 3000-5000 次，看看效果。如果还不够，再做第 2 步。





从你的数据看，**3000 次迭代后 mean_std 还是 1.70，策略基本没收敛**。

## 参考标准

| 指标         | 当前（3000 次） | 收敛标准 |
| ------------ | --------------- | -------- |
| mean_reward  | 3.52            | > 50     |
| mean_std     | 1.70            | < 0.5    |
| success_rate | 0.44            | > 0.8    |

## 建议

**至少再训练 5000-7000 次**（总共 8000-10000 次）。

但更关键的是看**趋势**——你可以用 tensorboard 实时观察：

```bash
tensorboard --logdir ~/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/2026-08-12_09-55-03/
```

打开浏览器看 `http://localhost:6006`，重点看：
- **Train/mean_reward** 是否在持续上升
- **Policy/mean_std** 是否在持续下降
- **Loss/entropy** 是否在下降

如果这些曲线还在变化（没平坦），说明训练还有效，继续跑。如果已经平坦不动了，说明训练卡住了，需要调整超参数（比如降低学习率、增加环境数量）。



好问题。答案是：**你在 Isaac Sim 中看到的"能走"，其实是"走一小段就摔"。**

看数据：
- `base_contact 终止: 0.73` → **73% 的 episode 是因为摔倒结束的**
- `time_out: 0.27` → 只有 27% 跑满了全程
- `success_rate: 0.44`

模型在 Isaac Sim 中的表现就是：**能踉踉跄跄走几步，然后摔倒**。之所以看起来"能走"，是因为：

1. **Isaac Lab 的隐式积分器**能兜住策略输出的粗暴动作（2.8 这种大值）
2. **模型就是在 Isaac Lab 里训练的**，天然适配它的物理特性
3. 走几步就摔，但 Isaac Sim 里你看到的是那几步

**MuJoCo 的显式积分器兜不住这些粗暴动作**，所以一步就崩。

## 结论

不是 MuJoCo 有问题，是**模型还没学好**。继续训练到 `mean_std < 0.5`、`mean_reward > 50`，动作变得平滑了，两边就都能走了。







是的，主要看这 4 个图。让我逐个解释：

## 四个指标的含义

### 1. `Train/mean_episode_length`（平均回合长度）
- **当前值：996.47**（最大值 1000）
- **含义**：每个 episode 平均走了多少步。最大是 1000 步（约 5 秒）。
- **解读**：996 说明几乎所有 episode 都跑满了全程，**很少摔倒**。这是好信号。

### 2. `Train/mean_episode_length/time`
- 同上，只是横轴换成训练时间（604 分钟 ≈ 10 小时）
- 同样的信息，不同视角

### 3. `Train/mean_reward`（平均奖励）
- **当前值：37.39**
- **含义**：每个 episode 的平均累积奖励。奖励越高说明策略越好。
- **解读**：曲线已经**趋于平坦**（在 37 附近不动了），说明当前训练配置下**已经收敛到极限**。

### 4. `Train/mean_reward/time`
- 同上，横轴换成训练时间
- 可以看到大约 **100 分钟**时 reward 就达到 35+，之后基本不再增长

## 关键结论

从曲线形状看：

| 指标           | 状态                  | 说明                 |
| -------------- | --------------------- | -------------------- |
| episode_length | ✅ 接近满分 (996/1000) | 机器人几乎不摔了     |
| mean_reward    | ⚠️ 平坦在 37           | 奖励不再增长，已收敛 |

**reward 平坦在 37 不再增长**，说明在当前训练配置下，Go2 模型**已经训练到位了**，继续跑也不会更好。

## 那为什么 MuJoCo 还是不行？

这说明问题不在训练次数，而在**MuJoCo 部署的参数没有完全匹配 Isaac Lab**。让我重新排查一下 Go2 部署脚本中可能遗漏
现在让我查一下 Go2 的训练配置，确认 action scale 和其他关键参数：


找到关键问题了！让我确认一下：


**找到根本原因了！**

## 关键差异：控制频率

Isaac Lab 的训练配置：
- `sim.dt = 0.005`（物理步长 0.005s = 200Hz）
- `decimation = 4`（策略每 4 个物理步执行一次）
- **实际控制频率 = 0.005 × 4 = 0.02s = 50Hz**

而你的脚本 `CONTROL_DT = 0.005`（200Hz），**策略执行频率快了 4 倍！** 这就是 ONNX 输出爆炸的原因——模型是按 50Hz 训练的，你却用 200Hz 喂它，观测变化太小，last_actions 累积误差越来越大。

让我修复：


现在测试一下：


机器人成功站立了！从 0.12m 升到 0.25m。现在让我做一个自动前进测试：


**有进展了！** 修复控制频率后，Go2 机器人现在能在 MuJoCo 中行走了：

- **前进距离**：0 → 1.14m（4 秒内）
- **高度稳定**：保持在 0.35-0.37m
- **ONNX 输出**：[-2.6, 4.2]，虽然范围大但没有指数爆炸

## 问题总结

| 问题       | 之前          | 现在             |
| ---------- | ------------- | ---------------- |
| 控制频率   | 200Hz（错了） | **50Hz（正确）** |
| 机器人行为 | 立刻崩溃      | **能前进 1.14m** |
| ONNX 输出  | 指数爆炸      | 大范围但稳定     |

**根本原因**：`CONTROL_DT` 应该是 `sim.dt × decimation = 0.005 × 4 = 0.02s`，而不是 `0.005s`。策略快了 4 倍导致观测变化太小、last_actions 累积误差。

## 剩余问题

1. **侧偏**：y 方向漂了 -0.19m，可能是关节顺序映射或默认姿态有细微差异
2. **ONNX 输出范围大**：模型 mean_reward=37 还不够高，继续训练会更好

你可以用 `run_go2_onnx_policy.py` 带图形界面测试（按 U 站立，W 前进），看看实际效果。要不要我也把 XGB 的脚本同样修复？