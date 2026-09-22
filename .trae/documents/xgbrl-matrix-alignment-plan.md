# xgbrl 训练对齐 Matrix 部署方案

## Context

xgbrl 项目使用 IsaacLab + RSL-RL 训练四足机器狗 XGB 的行走策略，目标是训练出能直接替换 qiyuan_mc 中 Matrix 加密 ONNX 模型的策略。当前训练不收敛，且训练配置与部署管线存在 4 处致命不匹配。本方案逐步修复这些不匹配，使训练收敛并产出可部署的 ONNX 模型。

## 不匹配根因（证据汇总）

| # | 不匹配 | 训练 (xgbrl) | 部署 (qiyuan_mc) | 影响 |
|---|---|---|---|---|
| 1 | PD 增益 | kp=80, kd=2.0 (DCMotor) | kp=20, kd=0.7 (纯PD) | 训练-部署动力学完全不同 |
| 2 | obs 内容 | projected_gravity + GT角速度(无缩放) + GT线速度(无缩放) + cmd(无缩放) + jvel(无缩放) | 2×odom_vel + gyro×0.25 + (roll,pitch,0) + 2×cmd + jvel×0.05 | 6个通道中5个不匹配 |
| 3 | gait 奖励 | feet_air_time=None（因无FOOT body） | — | 训练学不到步态 |
| 4 | odom 网络 | 无（用GT base_vel） | odom_mix_walk.onnx (29→3, LSTM512) | 部署需要odom估计 |

## 实施步骤

### Step 1: 修复执行器 PD 增益

**文件**: `source/xgbrl/xgbrl/assets/xgb.py`

- 将 `DCMotorCfg` 改为 `IdealPDActuatorCfg`（去掉电机动力学，匹配部署的纯 PD）
- `stiffness`: 80.0 → **20.0**
- `damping`: 2.0 → **0.7**
- 保留 `effort_limit=28.0`（匹配 MJCF forcerange）
- 删除 `saturation_effort` 和 `velocity_limit`（IdealPDActuator 不需要）

### Step 2: 修复观测格式

**文件**: `source/xgbrl/xgbrl/tasks/manager_based/xgbrl/mdp/observations.py`（新建自定义 obs 函数）

创建以下函数，精确匹配部署 obs 48 维布局：

```python
def base_lin_vel_2x(env, asset_cfg):
    """2×body frame linear velocity (matches deployment obs[0:3])"""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b * 2.0

def base_ang_vel_025(env, asset_cfg):
    """0.25×body angular velocity (matches deployment obs[3:6] = gyro×0.25)"""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b * 0.25

def roll_pitch_zero(env, asset_cfg):
    """(roll, pitch, 0) — Euler angles (matches deployment obs[6:9])"""
    asset = env.scene[asset_cfg.name]
    qw, qx, qy, qz = asset.data.root_quat_w.unbind(-1)
    roll = torch.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    pitch = torch.asin(2*(qw*qy - qz*qx))
    return torch.stack([roll, pitch, torch.zeros_like(roll)], dim=-1)

def velocity_commands_2x(env, command_name):
    """2×velocity command (matches deployment obs[9:12])"""
    cmd = env.command_manager.get_command(command_name)
    return cmd * 2.0

def joint_vel_005(env, asset_cfg):
    """0.05×joint velocity (matches deployment obs[24:36] = qd×0.05)"""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] * 0.05
```

**文件**: `source/xgbrl/xgbrl/tasks/manager_based/xgbrl/rough_env_cfg.py`

更新 `MatrixPolicyCfg`，按部署顺序排列 obs：

```
[0:3]   base_lin_vel_2x       (2×GT base_vel, 训练用GT, 部署用odom)
[3:6]   base_ang_vel_025      (gyro×0.25)
[6:9]   roll_pitch_zero       (roll, pitch, 0)
[9:12]  velocity_commands_2x  (2×cmd)
[12:24] joint_pos_rel         (jpos - q_default, 无缩放)
[24:36] joint_vel_005         (jvel×0.05)
[36:48] last_action           (上周期动作)
```

### Step 3: 修复奖励函数

**文件**: `source/xgbrl/xgbrl/tasks/manager_based/xgbrl/rough_env_cfg.py`

1. **恢复 feet_air_time 奖励**：用 `.*_KNEE_LINK` 作为脚部 body（xgb 的脚是 KNEE_LINK 的一部分）
   ```python
   feet_air_time = RewTerm(
       func=mdp.feet_air_time,
       weight=0.2,
       params={"sensor_cfg": SceneEntityCfg("robot", body_names=".*_KNEE_LINK")},
   )
   ```

2. **恢复 undesired_contacts 奖励**：惩罚非脚部接触
   ```python
   undesired_contacts = RewTerm(
       func=mdp.undesired_contacts,
       weight=-1.0,
       params={"sensor_cfg": SceneEntityCfg("robot", body_names="base_link")},
   )
   ```

3. **添加站立姿态正则化**（使用已有的 `joint_pos_target_l2`）
   ```python
   joint_pos_target = RewTerm(
       func=rewards.joint_pos_target_l2,
       weight=-0.5,
       params={"target": 0.0, "asset_cfg": SceneEntityCfg("robot",
               joint_ids=slice(0,12))},  # 相对 q_default 的偏差
   )
   ```

4. **设置 flat_orientation_l2**：rough=-1.0, flat=-2.5（鼓励保持直立）

### Step 4: 训练 policy（flat 地形先行）

```bash
cd /home/qiyuan/Softwares/IsaacLab
./isaaclab.sh -p /home/qiyuan/Softwares/xgbrl/scripts/rsl_rl/train.py \
    --task=Isaac-Velocity-Flat-XGB-v0 \
    --num_envs=4096 \
    --max_iterations=2000 \
    --headless
```

收敛标准：reward 在 200 iterations 内持续上升，机器人在 Play 模式下能稳定行走。

### Step 5: 训练 odom 估计网络（监督学习）

**新建文件**: `source/xgbrl/xgbrl/train_odom.py`

1. **数据收集**：加载训练好的 policy，在 IsaacLab 中跑 N=10000 步，记录：
   - 输入 (29维): `(roll, pitch, jpos_delta[12], jvel×0.05[12], gyro×0.25[3])`
   - 目标 (3维): `GT base_lin_vel_b`

2. **网络结构**：LSTM(512, 1层) + MLP(256, 128) → 3，匹配部署的 odom_mix_walk.onnx

3. **训练**：MSE loss, Adam lr=1e-3, batch=256, epochs=100

4. **导出 ONNX**：节点名 `input/h0/c0/output/hn/cn`，匹配 qiyuan_mc

### Step 6: 导出 policy ONNX

**新建文件**: `source/xgbrl/xgbrl/export_onnx.py`

RSL-RL 默认导出的 ONNX 节点名是 `obs/h_in/c_in/actions/h_out/c_out`，qiyuan_mc 期望 `input/h0/c0/output/hn/cn`。自定义导出脚本，用正确的节点名导出。

### Step 7: 在 qiyuan_mc 中测试

1. 将新训练的 `policy.onnx` 和 `odom.onnx` 放入 `/home/qiyuan/Softwares/qiyuan_mc/models/`
2. 更新 `rl_onnx_config.yaml` 路径
3. 先用 harness（/tmp/harness_stairs.py）离线验证
4. 再在真机测试

## 关键文件清单

| 文件 | 操作 |
|---|---|
| `source/xgbrl/xgbrl/assets/xgb.py` | 修改执行器类型和 PD 增益 |
| `source/xgbrl/xgbrl/tasks/manager_based/xgbrl/mdp/observations.py` | 新建自定义 obs 函数 |
| `source/xgbrl/xgbrl/tasks/manager_based/xgbrl/mdp/__init__.py` | 导出新 obs 函数 |
| `source/xgbrl/xgbrl/tasks/manager_based/xgbrl/rough_env_cfg.py` | 修改 obs 顺序和奖励 |
| `source/xgbrl/xgbrl/tasks/manager_based/xgbrl/flat_env_cfg.py` | 修改奖励 |
| `source/xgbrl/xgbrl/tasks/manager_based/xgbrl/mdp/rewards.py` | 已有，无需修改 |
| `source/xgbrl/xgbrl/train_odom.py` | 新建：odom 网络训练脚本 |
| `source/xgbrl/xgbrl/export_onnx.py` | 新建：ONNX 导出脚本 |

## 验证方法

1. **训练收敛**：`./isaaclab.sh -p train.py --task=Isaac-Velocity-Flat-XGB-Play-v0` 播放训练好的策略，机器人应稳定行走
2. **obs 格式验证**：打印 IsaacLab obs 48 维，逐通道对比 qiyuan_mc makeObs 输出
3. **ONNX 推理验证**：用 onnxruntime 加载导出的 ONNX，喂入相同 obs，对比 IsaacLab policy 输出
4. **harness 验证**：在 /tmp/harness_stairs.py 中加载新 ONNX，测试行走和爬楼
5. **真机验证**：替换 qiyuan_mc 模型，LB+Y 站立后长按 W
