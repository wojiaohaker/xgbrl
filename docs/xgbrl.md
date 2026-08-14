一、重新训练

xgbrl 的 gym 注册没有被加载。让我查一下 Isaac Lab 如何发现外部任务包：


找到了！xgbrl 有自己的训练脚本（第 43 行有 `import xgbrl.tasks`），要用它而不是 IsaacLab 的：

```bash
cd ~/Softwares/xgbrl && ~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-XGB-v0 --num_envs=4096
```

IsaacLab 的 `train` 命令不会自动加载 xgbrl 包，所以 gym 注册表里没有 XGB 环境。xgbrl 自己的脚本会先 `import xgbrl.tasks` 触发注册。



```bash
cd ~/Softwares/xgbrl && ~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task=Isaac-Velocity-Flat-XGB-Play-v0 --num_envs=1 --load_run=2026-08-14_14-31-23 --checkpoint=/home/qiyuan/Softwares/xgbrl/logs/rsl_rl/xgb_flat/2026-08-14_14-31-23/model_4999.pt --visualizer kit
```

如果想看最新的（3000 次迭代），用上面的命令。训练完成后可以换成 `model_5000.pt`。





一、训练命令

```
~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-XGB-v0 --num_envs=4096


cd ~/Softwares/xgbrl && ~/Softwares/IsaacLab/isaaclab.sh train --rl_library rsl_rl --task=Isaac-Velocity-Flat-XGB-v0 --num_envs=4096
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



四、关键脚本

## 三个脚本的作用

### 1. `xgb.py` — 机器人物理配置

**定义 XGB 机器人在仿真器中的物理属性：**

| 配置项           | 值                         | 含义           |
| ---------------- | -------------------------- | -------------- |
| `asset_path`     | `xgb.xml`                  | MJCF 模型路径  |
| `stiffness`      | 20.0                       | PD 控制器 Kp   |
| `damping`        | 0.7                        | PD 控制器 Kd   |
| `effort_limit`   | 28.0 Nm                    | 关节力矩上限   |
| `velocity_limit` | 30.0 rad/s                 | 关节速度上限   |
| `init_state.pos` | (0, 0, 0.32)               | 初始高度 32cm  |
| `joint_pos`      | ABAD=0, HIP=0.8, KNEE=-1.5 | 站立姿态关节角 |

**作用**：告诉 Isaac Lab "XGB 长什么样、物理参数是多少"。

---

### 2. `flat_env_cfg.py` — 训练环境配置

**定义 RL 训练环境的规则：**

| 配置项                                      | 含义               |
| ------------------------------------------- | ------------------ |
| `terrain_type = "plane"`                    | 平地训练（无起伏） |
| `rewards.flat_orientation_l2.weight = -2.5` | 惩罚机身倾斜       |
| `num_envs = 4096`                           | 并行 4096 个环境   |
| `episode_length_s`                          | episode 时长       |

**两个类：**
- `XgbFlatEnvCfg` — 训练用（有随机化、有终止条件）
- `XgbFlatEnvCfg_PLAY` — 播放用（无随机化、无终止、固定速度 0.5m/s 前进）

**作用**：告诉 Isaac Lab "训练规则是什么、奖励怎么算"。

---

### 3. `rsl_rl_ppo_cfg.py` — RL 算法配置

**定义 PPO 算法的超参数：**

| 配置项               | 值            | 含义               |
| -------------------- | ------------- | ------------------ |
| `max_iterations`     | 10000         | 训练迭代次数       |
| `num_steps_per_env`  | 24            | 每次收集 24 步数据 |
| `learning_rate`      | 1e-4          | 学习率             |
| `actor.hidden_dims`  | [128,128,128] | Actor 网络结构     |
| `critic.hidden_dims` | [128,128,128] | Critic 网络结构    |
| `init_std`           | 0.5           | 动作探索初始标准差 |
| `std_type`           | "scalar"      | std 参数化方式     |
| `gamma`              | 0.99          | 折扣因子           |
| `entropy_coef`       | 0.02          | 熵正则化系数       |
| `max_grad_norm`      | 1.0           | 梯度裁剪           |

**两个类：**
- `XgbRoughPPORunnerCfg` — 粗糙地形用（大网络 512→256→128）
- `XgbFlatPPORunnerCfg` — 平地用（小网络 128→128→128，继承并覆盖）

**作用**：告诉 RSL_RL "PPO 怎么训练、网络多大、学习率多少"。

---

## 三者关系

```
xgb.py (机器人)
    ↓ 被引用
flat_env_cfg.py (环境)
    ↓ 被引用
rsl_rl_ppo_cfg.py (算法)
    ↓
train.py 启动训练
```

- **xgb.py** → 定义"谁在学"
- **flat_env_cfg.py** → 定义"在哪学、学什么"
- **rsl_rl_ppo_cfg.py** → 定义"怎么学"





五、训练流程

## 脚本执行流程

```
命令行
  ↓
train.py
```

### 第 1 步：解析参数（L67-88）

```
--task=Isaac-Velocity-Flat-XGB-v0  →  找到任务配置
--num_envs=4096                    →  并行环境数
```

### 第 2 步：导入任务注册（L55）

```python
import xgbrl.tasks  # 触发 __init__.py → gym.register()
```

这会执行 `xgbrl/tasks/manager_based/xgbrl/__init__.py`，注册 4 个环境：
- `Isaac-Velocity-Flat-XGB-v0`
- `Isaac-Velocity-Flat-XGB-Play-v0`
- `Isaac-Velocity-Rough-XGB-v0`
- `Isaac-Velocity-Rough-XGB-Play-v0`

### 第 3 步：Hydra 加载配置（L124）

```python
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
```

根据任务名 `Isaac-Velocity-Flat-XGB-v0`，从注册信息中找到：
- `env_cfg` ← `flat_env_cfg:XgbFlatEnvCfg`
- `agent_cfg` ← `rsl_rl_ppo_cfg:XgbFlatPPORunnerCfg`

### 第 4 步：启动仿真器（L127）

```python
with launch_simulation(env_cfg, args_cli):
```

- 启动 Isaac Sim
- 创建 PhysX 物理场景
- 加载 XGB 机器人（`xgb.py` → `xgb.xml`）
- 复制 4096 个环境

### 第 5 步：覆盖 CLI 参数（L129-133）

```python
env_cfg.scene.num_envs = 4096        # 命令行覆盖
agent_cfg.max_iterations = 10000     # 配置文件中的值
```

### 第 6 步：创建环境（L192）

```python
env = gym.make("Isaac-Velocity-Flat-XGB-v0", cfg=env_cfg)
```

这会：
- 创建地形（平地）
- 设置奖励函数（`flat_env_cfg.py` 中的 rewards）
- 设置观测空间
- 设置终止条件

### 第 7 步：包装环境（L219）

```python
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
```

把 Isaac Lab 环境包装成 RSL_RL 需要的向量环境接口。

### 第 8 步：创建 PPO Runner（L222-223）

```python
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
```

根据 `rsl_rl_ppo_cfg.py` 创建：
- **Actor 网络**：[128, 128, 128]，输出 12 维动作
- **Critic 网络**：[128, 128, 128]，输出状态价值
- **PPO 算法**：lr=1e-4, gamma=0.99, entropy_coef=0.02

### 第 9 步：保存配置（L241-242）

```python
dump_yaml("logs/rsl_rl/xgb_flat/2026-08-14_XX-XX-XX/params/env.yaml", env_cfg)
dump_yaml("logs/rsl_rl/xgb_flat/2026-08-14_XX-XX-XX/params/agent.yaml", agent_cfg)
```

### 第 10 步：开始训练（L246）

```python
runner.learn(num_learning_iterations=10000, init_at_random_ep_len=True)
```

**训练循环：**
```
for iteration in range(10000):
    1. 收集数据：4096 个环境 × 24 步 = 98304 条经验
    2. 计算优势函数（GAE）
    3. PPO 更新：
       - Actor 更新（策略梯度）
       - Critic 更新（价值回归）
    4. 记录 TensorBoard 指标
    5. 每 50 次迭代保存模型（model_50.pt, model_100.pt, ...）
```

### 第 11 步：训练结束（L249）

```python
env.close()  # 关闭仿真器
```

---

## 总结流程图

```
命令行参数
    ↓
解析 --task → 找到 gym.register() 的配置
    ↓
加载 env_cfg (flat_env_cfg.py) + agent_cfg (rsl_rl_ppo_cfg.py)
    ↓
启动 Isaac Sim → 加载 xgb.py → 创建 4096 个环境
    ↓
创建 PPO Runner（Actor/Critic 网络）
    ↓
训练循环 10000 次：
    收集数据 → 计算优势 → 更新网络 → 保存模型
    ↓
训练完成 → 关闭仿真器
```





六、崩溃

训练跑到 4967 次了，但最后崩溃了。问题变了：

| 指标           | 值      | 问题                             |
| -------------- | ------- | -------------------------------- |
| value loss     | NaN     | 价值网络崩溃                     |
| surrogate loss | NaN     | 策略梯度崩溃                     |
| entropy        | -61.64  | 应该是正数，负值说明概率计算异常 |
| action std     | 0.00    | std 完全坍缩                     |
| **观测值**     | **NaN** | **环境返回了 NaN**               |

**根本原因**：PhysX GPU 内存不足（`foundLostPairsCapacity` 错误），导致物理仿真产生 NaN，污染了观测值。

## 解决方案

**减少并行环境数**，降低 GPU 内存压力：

```bash
rm -rf ~/Softwares/xgbrl/logs/rsl_rl/xgb_flat/

cd ~/Softwares/xgbrl && ~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-XGB-v0 --num_envs=2048
```

从 4096 减到 2048，应该能避免 PhysX 内存溢出。



七、大佬修复

```bash
cd ~/Softwares/Mujoco330/demo_xgb && ~/Softwares/IsaacLab/env_isaaclab/bin/python run_onnx_policy_fix.py /home/qiyuan/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/2026-08-12_09-55-03/exported/policy.onnx --mjcf /home/qiyuan/Softwares/Matrix/src/robot_mujoco/zsibot_robots/xgb/scene_terrain_flat.xml
```



八、我的问题

```bash
cd ~/Softwares/Mujoco330/demo_xgb && ~/Softwares/IsaacLab/env_isaaclab/bin/python run_onnx_policy.py /home/qiyuan/Softwares/IsaacLab/logs/rsl_rl/xgb_flat/2026-08-12_09-55-03/exported/policy.onnx --mjcf /home/qiyuan/Softwares/Matrix/src/robot_mujoco/zsibot_robots/xgb/scene_terrain_flat.xml
```



九、5000 play

训练已完成 5000 次迭代！让我确认一下 play 命令的正确格式：


训练已完成 5000 次迭代。用最新模型 `model_4999.pt` 来 play：


模型版本不匹配 — checkpoint 用旧版 rsl_rl 保存的（`std_param`），当前版本期望 `log_std_param`。让我修复这个兼容性问题：


只需将 `distribution.std_param` 重命名为 `distribution.log_std_param`：


修复完成，用修正后的 checkpoint 重新 play：


模型加载成功，Play 模式已在运行。Isaac Sim 窗口应该已经打开，你可以在里面看到机器人的行走效果。

训练信息总结：
- **训练轮次**：5000 次迭代（model_4999.pt）
- **最终奖励**：可查看 TensorBoard 日志
- **观测空间**：48 维（base_lin_vel 3 + base_ang_vel 3 + projected_gravity 3 + velocity_commands 3 + joint_pos 12 + joint_vel 12 + actions 12）

注意：
1. Play 模式启用了 termination（`time_out` + `base_contact`），机器人摔倒后会自动重置
2. 有 `push_robot` 扰动（每 10-15 秒随机推一次），可以观察策略的抗干扰能力
3. 速度命令由环境随机生成，机器人会尝试跟踪不同的目标速度

你可以在 Isaac Sim 窗口中观察效果。如果想导出 ONNX 用于 MuJoCo 部署，play 结束后会在日志目录生成 `exported/policy.onnx`。



Play 模式已经在后台运行了，Isaac Sim 窗口应该已经弹出。如果你想自己重新运行，命令是：

```bash
cd ~/Softwares/xgbrl && ~/Softwares/IsaacLab/isaaclab.sh play --rl_library rsl_rl --task=Isaac-Velocity-Flat-XGB-v0 --num_envs=1 --checkpoint=logs/rsl_rl/xgb_flat/2026-08-14_11-26-19/model_4999_fixed.pt --viz kit
```

如果想看 TensorBoard 训练曲线：

```bash
tensorboard --logdir ~/Softwares/xgbrl/logs/rsl_rl/xgb_flat/2026-08-14_11-26-19
```



十、训练配置

## 影响训练质量的核心因素

根据你的 XGB 训练数据和配置，以下是关键影响因素：

---

### 1. **奖励函数设计**（最重要）

| 奖励项                 | 当前权重 | 作用         | 问题                       |
| ---------------------- | -------- | ------------ | -------------------------- |
| `track_lin_vel_xy_exp` | 1.5      | 跟踪速度命令 | ✅ 核心奖励                 |
| `track_ang_vel_z_exp`  | 0.75     | 跟踪转向命令 | ✅ 合理                     |
| `flat_orientation_l2`  | -2.5     | 保持水平     | ⚠️ 权重偏高，初期阻碍学习   |
| `lin_vel_y_l2`         | **-0.5** | 惩罚侧向移动 | ✅ 已降低（之前 -2.0 过重） |
| `ang_vel_z_l2`         | **-0.3** | 惩罚转向     | ✅ 已降低（之前 -1.0 过重） |
| `action_rate_l2`       | -0.01    | 平滑动作     | ✅ 合理                     |

**关键原则**：先让机器人学会站立（降低惩罚），再逐步增加约束。

---

### 2. **速度命令范围**

```python
# 当前配置（已修复）
lin_vel_x = (0.0, 1.5)    # 前进 0-1.5 m/s
lin_vel_y = (-0.5, 0.5)   # 侧向 ±0.5 m/s
ang_vel_z = (-1.5, 1.5)   # 转向 ±1.5 rad/s
```

**问题**：之前设置 `vx∈[0,3]`, `wz∈[-3,3]`，超出 XGB 物理极限，导致：
- 策略尝试跟踪不可能的命令 → 产生 NaN
- 训练崩溃后无法恢复

---

### 3. **PPO 超参数**

| 参数                | 当前值   | 影响           | 建议范围     |
| ------------------- | -------- | -------------- | ------------ |
| `learning_rate`     | **1e-4** | 学习速度       | 1e-4 ~ 3e-4  |
| `num_steps_per_env` | **48**   | Rollout 样本量 | 24 ~ 64      |
| `entropy_coef`      | **0.02** | 探索强度       | 0.01 ~ 0.02  |
| `num_mini_batches`  | 4        | 梯度更新次数   | 4 ~ 8        |
| `gamma`             | 0.99     | 未来奖励的折扣 | 0.99 ~ 0.995 |
| `max_iterations`    | **8000** | 训练总轮数     | 5000 ~ 10000 |

**关键**：`learning_rate` 过高会导致训练不稳定（你之前的 NaN 崩溃就是因此）。

---

### 4. **机器人模型参数**

```python
# xgb.py
effort_limit = 28.0      # 力矩上限
velocity_limit = 30.0    # 速度上限
stiffness = 20.0         # PD 控制 Kp
damping = 0.7            # PD 控制 Kd
action_scale = 0.25      # 动作缩放
```

**影响**：
- `stiffness/damping` 决定策略能输出的动作范围
- `action_scale` 太小 → 动作受限；太大 → 动作过激
- 当前 `KP=20, KD=0.7` 匹配 Matrix，✅ 合理

---

### 5. **终止条件**

```python
terminations.base_contact  # base_link 触地 → 终止
terminations.time_out      # 超时（默认 20s）→ 终止
```

**问题**：如果机器人 70% 的时间都在摔倒（你之前的数据），说明：
- 初始姿态不稳定
- 或者策略没学会保持平衡

---

### 6. **环境数量**

```bash
--num_envs 4096  # 当前使用
```

**影响**：
- 太少（<1024）→ 样本多样性不足，收敛慢
- 太多（>8192）→ 显存不足，训练慢
- **4096 是合理值** ✅

---

### 7. **Domain Randomization（域随机化）**

当前启用的随机化：
- `add_base_mass` - 基座质量随机
- `base_com` - 质心偏移随机
- `base_external_force_torque` - 外力扰动
- `push_robot` - 随机推力（每 10-15s）

**作用**：提高策略鲁棒性，但过多随机化会阻碍初期学习。

---

### 8. **NaN 处理**（你之前遇到的问题）

```python
# utils.py - check_nan
if nan_envs.any():
    dones[nan_envs] = True  # 终止有 NaN 的环境
```

**问题**：之前替换 NaN 为 0 后继续训练，导致策略损坏。现在改为终止该环境，避免污染训练。

---

## 你的训练数据对比

| 训练                    | 迭代数 | 峰值 reward | 最终 reward | 峰值 length | 最终 length |
| ----------------------- | ------ | ----------- | ----------- | ----------- | ----------- |
| **2026-08-14_11-26-19** | 5000   | 5.01        | **-inf**    | 981         | 394         |
| **2026-08-14_14-02-39** | 314    | 4.60        | 1.78        | 952         | 240         |

**分析**：
- 第一次训练在 step 1932 崩溃，之后再也没恢复
- 第二次训练（新配置）才 314 步，还在早期，峰值 4.60 说明有改善

---

## 建议

1. **继续观察第二次训练**（2026-08-14_14-02-39），看能否稳定超过 5.0
2. 如果 reward 再次崩溃，考虑进一步降低 `learning_rate` 到 `5e-5`
3. 如果 `base_contact` 摔倒率仍然 >50%，考虑降低 `flat_orientation_l2` 权重到 `-1.5`
