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
