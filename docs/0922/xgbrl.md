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



是的，**完全正常**！从截图看：

1. ✅ 四条腿完全对称（FAR、FBL、RAR、RBL 姿态一致）
2. ✅ 关节结构正确（ABAD → HIP → KNEE 层级正确）
3. ✅ 模型没有左右不对称的问题

**结论：MJCF 模型本身没问题，USD 转换也没问题。**

之前训练时左前腿异常的原因是 **PD 增益太低**（stiffness=20, damping=0.7），不足以维持目标关节位置，导致 reset 后关节位置漂移不对称。

需要在 `flat_env_cfg.py` 和 `rough_env_cfg.py` 里提高 PD 增益。根据你之前的经验（Kp=150, Kd=2.0），我来修改：


模型完全正常！问题确认是 **PD 增益太低**。

当前 `stiffness=20, damping=0.7` 是 Matrix 部署时的值，但在 Isaac Lab RL 训练中太低，无法维持站立姿态。

修改 `xgb.py` 提高 PD 增益：


已修改 PD 增益：`stiffness: 20→80, damping: 0.7→2.0`

**需要重新训练**，因为旧模型是用低增益训练的，和新增益不兼容。

训练命令：
```bash
cd ~/Softwares/xgbrl && ~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-XGB-v0 --num_envs=4096
```





让我先看看当前模型和配置的状态：


让我检查一下当前的模型和配置状态：


让我先检查当前的配置状态：


从截图看，机器人在 Isaac Sim 中姿态异常。让我先检查当前配置和训练状态：


从截图看，机器人在 Isaac Sim 中姿态不对。让我检查一下当前训练状态和配置
看到了！你在用训练好的模型（model_4999.pt）做 play 测试，但机器人姿态不对。让我检查一下训练配置
让我检查训练日志，看看训练是否正常完成：


从截图看，机器人在 Isaac Sim play 模式下姿态完全不对。根据之前的经验记录，你要求的是 **Kp=150, Kd=2.0**，但当前 `xgb.py` 里是 `stiffness=80.0`。

让我修复 PD 增益：


找到问题了！看 MJCF 文件的关节顺序：

**MJCF 物理顺序**（按腿分组）：
```
FAR(FR): ABAD, HIP, KNEE → 索引 0,1,2
FBL(FL): ABAD, HIP, KNEE → 索引 3,4,5  
RAR(RR): ABAD, HIP, KNEE → 索引 6,7,8
RBL(RL): ABAD, HIP, KNEE → 索引 9,10,11
```

**Isaac Lab 的 `joint_names_expr` 顺序**（按关节类型分组）：
```
.*_ABAD_JOINT → FAR, FBL, RAR, RBL (索引 0,3,6,9)
.*_HIP_JOINT  → FAR, FBL, RAR, RBL (索引 1,4,7,10)
.*_KNEE_JOINT → FAR, FBL, RAR, RBL (索引 2,5,8,11)
```

这意味着 Isaac Lab 训练时的关节顺序是 `[ABAD×4, HIP×4, KNEE×4]`，而不是 `[FR×3, FL×3, RR×3, RL×3]`。

但更重要的是，根据之前的经验记录，你要求的是 **Kp=150, Kd=2.0**，当前 `xgb.py` 里是 `stiffness=80.0`，PD 增益太低导致无法克服重力维持站立姿态。


等等，我刚才把 PD 增益改成了 150，但你的模型是用 stiffness=80 训练的。**PD 增益改变后需要重新训练**，否则训练和部署的动力学不匹配。

让我先改回来，然后分析
让我检查训练是否正常收敛：


找到根本原因了！训练日志显示：

- **`Loss/value`**: 从 0.01 爆炸到 **1025558542694089696204554240.00**（天文数字！）
- **`Episode_Reward/action_rate_l2`**: 从 -0.00 爆炸到 **-3,837,903.00**
- **`Train/mean_reward`**: 只从 -0.68 提升到 7.86（很低）

**训练崩溃了！** Value loss 和 action rate 都爆炸了，模型输出了极端动作，所以机器人四条腿都不对。

让我检查训练配置：


问题找到了！训练配置中 **`init_std: 0.8` 太高了**，导致：
1. 初始探索动作太大 → 机器人做极端动作
2. Value loss 爆炸到 10^30
3. Action rate 惩罚爆炸到 -380 万

根据之前的经验，需要降低 `init_std` 来避免训练崩溃。让我检查完整的奖励配置：


问题明确了！训练崩溃的根因是 **`init_std: 0.8` 太高**。让我找到训练配置文件并修复：

找到了！`init_std=0.8` 在 `rsl_rl_ppo_cfg.py` 第 21 行。根据经验记录，需要降低 `init_std` 来避免训练崩溃。






**问题总结：**

你的训练崩溃了！从 TensorBoard 日志可以看到：

| 指标                | 初始值 | 最终值         | 状态   |
| ------------------- | ------ | -------------- | ------ |
| `Train/mean_reward` | -0.68  | 7.86           | 很低   |
| `Loss/value`        | 0.01   | **10^30**      | 💥 爆炸 |
| `action_rate_l2`    | -0.00  | **-3,837,903** | 💥 爆炸 |

**根因：`init_std=0.8` 太高**
- 初始探索动作太大（joint offset 可达 ±0.8×0.25=±0.2 rad）
- 导致机器人做极端动作 → value function 无法预测 → value loss 爆炸
- 最终模型输出极端动作 → 四条腿姿态都不对

**已修复：**
- `init_std`: 0.8 → **0.4**（在 `rsl_rl_ppo_cfg.py`）

**下一步：重新训练**

```bash
cd /home/qiyuan/Softwares/xgbrl
~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-XGB-v0 --headless
```

新训练应该会更稳定，reward 会更高，机器人姿态也会正常。





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



二、play

```
~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint logs/rsl_rl/xgb_flat/2026-08-17_15-45-34/model_999.pt \
  --viz kit


~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint logs/rsl_rl/xgb_flat/2026-08-14_16-19-20/model_4999.pt \
  --visualizer kit
  
~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint logs/rsl_rl/xgb_flat/2026-08-14_16-19-20/model_4999.pt \
  --viz kit
  
~/Softwares/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task=Isaac-Velocity-Flat-XGB-Play-v0 \
  --num_envs=1 \
  --checkpoint logs/rsl_rl/xgb_flat/2026-08-14_17-55-21/model_999.pt \
  --viz kit
  

  
IsaacLabTutorial
~/Softwares/IsaacLab/isaaclab.sh -p scripts/skrl/play.py --task Template-Isaac-Lab-Tutorial-Direct-v0 --num_envs 10 --viz kit
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





## 什么是"观测"（Observation）

在强化学习运控中，**观测 = 策略网络的输入向量**。每个控制周期（2ms），控制器把所有感知信息打包成一个固定长度的浮点数组，喂给 ONNX 模型，模型输出 12 个关节动作。

```
观测 obs[48]  →  [ONNX 策略网络]  →  动作 action[12]
```

可以理解为：**观测就是机器人"看到"的全部世界**。策略网络是"大脑"，观测就是"眼睛+内耳+记忆"传给大脑的所有信号。

观测通常包含：
| 类别       | 含义                  | 类比                  |
| ---------- | --------------------- | --------------------- |
| 本体姿态   | 重力方向、角速度      | 内耳（平衡感）        |
| 基座速度   | 当前移动速度          | 本体感觉              |
| 速度指令   | 期望走多快            | "想走快一点"的意图    |
| 关节状态   | 各关节角度和速度      | 本体感觉（肌肉/肌腱） |
| 上一步动作 | 上一周期输出的 action | 短期记忆              |

---

## Matrix 的观测是什么样的

从之前的分析，Matrix 的 walk 策略（`policy_mix_walk`）观测是 **48 维**，结构如下：

| 索引     | 维度   | 内容                  | 数据来源                               |
| -------- | ------ | --------------------- | -------------------------------------- |
| [0:2]    | 3      | **projected_gravity** | IMU → 四元数 → 重力在本体坐标系的投影  |
| [3:5]    | 3      | **angular_velocity**  | IMU 陀螺仪直接读取                     |
| [6:8]    | 3      | **base_vel** (odom)   | **odom 模型推理输出**（不是直接测量）  |
| [9:11]   | 3      | **vel_cmd**           | 手柄/上位机下发的速度指令 (vx, vy, wz) |
| [12:23]  | 12     | **joint_pos_rel**     | 12 关节位置 − 默认站立位置（相对偏差） |
| [24:35]  | 12     | **joint_vel**         | 12 关节速度                            |
| [36:47]  | 12     | **last_action**       | 上一个控制周期的策略输出               |
| **合计** | **48** |                       |                                        |

### 关键推理流程（从 Matrix 日志还原）

```
每 2ms 一个周期：

  ① 接收 RobotState（UDP）
     → q[12], qd[12], quat[4], gyro[3], acc[3]

  ② Odom 推理（LSTM）
     input[29] → odom模型 → output[3] = base_vel

  ③ 拼观测 obs[48]
     [proj_grav(3), ang_vel(3), base_vel(3), vel_cmd(3),
      joint_pos_rel(12), joint_vel(12), last_action(12)]

  ④ Policy 推理（LSTM）
     obs[48] + h_in[512] + c_in[512] → policy模型 → actions[12]

  ⑤ PD 控制
     τ = Kp * (stand_pos + 0.25 * action - q) - Kd * qd

  ⑥ 发送 RobotCmd（UDP）
```

### Matrix 不同策略的观测维度

| 策略          | obs 维度 | 多出来的部分            |
| ------------- | -------- | ----------------------- |
| walk          | 48       | 标准观测                |
| jump/backflip | 49       | +1 相位变量             |
| balance_stand | 51       | +3 位置目标 (x, y, yaw) |
| tracking      | 52       | +4 目标位置/朝向        |
| IK            | 93       | +45 足端目标轨迹        |
| measured      | 169      | +121 长历史帧/力传感器  |

---

## 你的 Isaac Lab 训练 vs Matrix 的观测对比

| 索引    | 维度 | Isaac Lab 训练                | Matrix 部署           |
| ------- | ---- | ----------------------------- | --------------------- |
| [0:2]   | 3    | **base_lin_vel** (仿真器真值) | **projected_gravity** |
| [3:5]   | 3    | **base_ang_vel** (仿真器真值) | **angular_velocity**  |
| [6:8]   | 3    | **projected_gravity**         | **base_vel (odom)**   |
| [9:11]  | 3    | **velocity_commands**         | **velocity_commands** |
| [12:23] | 12   | **joint_pos_rel**             | **joint_pos_rel**     |
| [24:35] | 12   | **joint_vel_rel**             | **joint_vel**         |
| [36:47] | 12   | **last_action**               | **last_action**       |

**关键差异**：
- **顺序不同** — Isaac Lab 是 `lin_vel → ang_vel → gravity → cmd`，Matrix 是 `gravity → ang_vel → base_vel(odom) → cmd`
- **base_lin_vel vs base_vel(odom)** — 训练时用仿真器真值，部署时 Matrix 用 odom 模型估计值替代
- **内容基本一致** — 都是这 7 组信息，只是排列顺序和 base_vel 的来源不同

这就是为什么 qiyuan_mc 的 `buildObservation()` 需要严格对齐训练时的顺序，同时用 odom 估计值替代训练时的 `base_lin_vel` 真值。





让我验证新模型的观测顺序是否正确：


观测顺序验证通过，完全匹配 Matrix 格式：

```
1. projected_gravity  (3)   ← 重力投影
2. base_ang_vel       (3)   ← 角速度
3. base_lin_vel       (3)   ← 线速度（部署时用 odom 估计替代）
4. velocity_commands  (3)   ← 速度指令
5. joint_pos          (12)  ← 关节相对位置
6. joint_vel          (12)  ← 关节速度
7. actions            (12)  ← 上一步动作
8. height_scan        null  ← 已禁用
─────────────────────────────
总计                   48 维
```

**模型结构**：
- LSTM ✅：`obs[48] + h_in[512] + c_in[512] → actions[12] + h_out[512] + c_out[512]`
- 与 Matrix 的 walk 策略完全一致

现在可以把模型复制到 qiyuan_mc 使用了：

```bash
cp /home/qiyuan/Softwares/xgbrl/logs/rsl_rl/xgb_flat/2026-08-17_15-45-34/exported/policy.onnx \
   /home/qiyuan/Softwares/qiyuan_mc/models/policy.onnx
cp /home/qiyuan/Softwares/xgbrl/logs/rsl_rl/xgb_flat/2026-08-17_15-45-34/exported/policy.onnx.data \
   /home/qiyuan/Softwares/qiyuan_mc/models/policy.onnx.data
```

需要我帮你复制并测试运行吗？







好的，让我从实际运行的 Matrix 重新分析站立和趴下的实现。


好的，我从源码重新分析 Matrix mc_ctrl 的站立和趴下实现。


源码不在 robot_mc 目录下，让我找找实际源码位置：


没有源码文件，让我查看 build 目录和整个 Matrix 项目结构：


源码可能是通过 git 管理的。让我查找实际的源码文件：


源码不在 git 中。让我找实际的 mc_ctrl 二进制和库文件，以及可能的源码位置：


源码只有编译后的二进制文件。让我从运行日志、配置文件和二进制符号来分析：


非常好，日志信息很丰富。让我继续读取配置文件和分析二进制符号：


现在分析二进制中的 FSM 符号和 motion 配置：


让我查看 qiyuan_mc 中的 FSM 头文件来了解状态枚举定义：


现在我已经收集了足够的信息。让我综合分析。


## Matrix mc_ctrl 站立（U键）与趴下（Space键）分析

### 一、整体架构

mc_ctrl 采用**两层状态机**架构：

```
┌─────────────────── 主 FSM ───────────────────┐
│  PASSIVE → JOINT_PD → STANDUP → RL_MIX      │
│                          ↕                    │
│                   RL_BALANCE_STAND            │
│                                              │
│  STANDUP 内部包含一个 Motion 子状态机:        │
│  ┌─ Motion_State_RecoveryStand (核心)        │
│  │   ├── standUp()   ← 站立                 │
│  │   ├── lieDown()   ← 趴下                 │
│  │   ├── foldLegs()  ← 收腿                 │
│  │   ├── rollOver()  ← 翻身                 │
│  │   └── sideLying() ← 侧卧                 │
│  ├─ Motion_State_MotionMix (特殊动作)        │
│  └─ Motion_State_Greet (打招呼)              │
└──────────────────────────────────────────────┘
```

### 二、按键输入链路

```
键盘 U/Space
    ↓
MuJoCo GLFW 窗口 / 物理手柄
    ↓
Joystick 设备 (/dev/input/js*)
    ↓
mc_ctrl: GamepadReader (500Hz 周期任务)
    ↓
GamepadCommand → DesiredStateCommand::handleGamePadCommand()
    ↓
修改 param_control_mode / param_motion_mode
    ↓
ControlFSM::checkTransition() 检测状态跳转条件
    ↓
切换到目标 FSM 状态
```

日志确认：
```
[joystic] find 1 joystic
[PeriodicTask] Start GamepadReader (0 s, 2000000 ns)
```

### 三、站立过程（U 键）

**不是 ONNX 模型控制，是纯 PD 控制。**

#### 状态流转
```
PASSIVE → JOINT_PD → STANDUP(standUp) → RL_MIX
```

#### STANDUP 状态内部实现

`FSM_State_StandUp::run()` 调用 `Motion_State_RecoveryStand::standUp()`：

**阶段 1：增益渐进（ramp up）**
- 由 `standup_ramp_iter` 控制迭代次数
- PD 增益从 0 渐进到目标值：
  - ABAD/HIP/KNEE Kp = **20.0**（`FSM_RL_ABAD_Kp`）
  - Kd = **0.7**（`FSM_RL_Kd`）

**阶段 2：关节插值**
- 目标关节位置从当前姿态插值到站立姿态：
  ```yaml
  abad_stand_pos:  [0, 0, 0, 0]       # rad
  hip_stand_pos:   [0.8, 0.8, 0.8, 0.8]  # rad
  knee_stand_pos:  [-1.5, -1.5, -1.5, -1.5]  # rad
  ```
- 使用 `setJPosInterPts()` 设置插值起点和终点

**阶段 3：稳定等待（settle）**
- 由 `standup_settle_iter` 控制
- 保持站立姿态，等待机器人稳定

**阶段 4：跳转条件**
- 检查 `body_height` 是否达到阈值
- 日志：`body height is %f, Stand up`
- 条件满足 → 跳转到 `RL_MIX`，ONNX 策略接管

#### 控制公式
```
τ = Kp × (q_des - q) + Kd × (qd_des - qd)

其中:
  q_des  = 站立目标位置（插值中）
  qd_des = 0（目标速度为零）
  Kp, Kd = 渐进增益
```

### 四、趴下过程（Space 键）

**同样不是 ONNX 模型控制，是纯 PD 控制。**

#### 状态流转
```
RL_MIX → STANDUP(lieDown/foldLegs) → PASSIVE
```

#### 跳转条件
从日志中的 transition 条件可知：
```
transition condition is not met, can't transition from rlmix to liedown
```
说明从 RL_MIX 到趴下需要满足特定条件（body_height 足够低）。

#### STANDUP 状态内部实现

`Motion_State_RecoveryStand` 根据意图执行 `lieDown()` 或 `foldLegs()`：

**关节目标：站立 → 趴下姿态**
```yaml
abad_liedown_pos:  [0, 0, 0, 0]
hip_liedown_pos:   [1.4, 1.4, 1.4, 1.4]    # 比站立(0.8)更大
knee_liedown_pos:  [-2.4, -2.4, -2.4, -2.4] # 比站立(-1.5)更弯
```

**过程**：
1. 关节位置从站立插值到趴下姿态
2. `fold_ramp_iter` 控制 PD 增益渐进
3. `fold_settle_iter` 等待稳定
4. 日志：`body height is %f, Folding legs`
5. 条件检查：`body height is still too low %f, or UpsideDown Folding legs`
6. 高度足够 → 跳转到 `PASSIVE`（电机阻尼）

### 五、关键配置参数

| 参数                  | 值         | 说明                 |
| --------------------- | ---------- | -------------------- |
| `enable_standup`      | 1          | 允许站立             |
| `enable_foldleg`      | 1          | 允许收腿趴下         |
| `FSM_RL_ABAD_Kp`      | 20.0       | RL 模式 ABAD Kp      |
| `FSM_RL_HIP_Kp`       | 20.0       | RL 模式 HIP Kp       |
| `FSM_RL_KNEE_Kp`      | 20.0       | RL 模式 KNEE Kp      |
| `FSM_RL_Kd`           | 0.7        | RL 模式 Kd           |
| `FSM_passive_Kd`      | 3.0        | 被动模式阻尼         |
| `FSM_jointPD_Kp`      | 80         | JOINT_PD 模式 Kp     |
| `FSM_jointPD_Kd`      | 1.0        | JOINT_PD 模式 Kd     |
| `standup_ramp_iter`   | (内部变量) | 站立增益渐进迭代次数 |
| `standup_settle_iter` | (内部变量) | 站立稳定等待迭代次数 |
| `fold_ramp_iter`      | (内部变量) | 趴下增益渐进迭代次数 |
| `fold_settle_iter`    | (内部变量) | 趴下稳定等待迭代次数 |

### 六、完整 FSM 状态初始化顺序

从日志可以看到启动时的初始化：
```
Initialized FSM state: PASSIVE
Initialized FSM state: JOINT_PD
Initialized FSM state: STANDUP
Initialized Motion state          ← Motion_State_RecoveryStand
Initialized FSM state: JOINT_FREE
Initialized FSM state: SAFE_PROTECT
Initialized FSM state: RL_MLP
Initialized FSM state: RL_MIX     ← 主 RL 策略
Initialized FSM state: RL_BALANCE_STAND
Initialized FSM state: RL_WalkPos
Initialized FSM state: RL_IK
Initialized FSM state: JOINT_LOCK
Initialized FSM state: POS_CONTROL
```

### 七、总结

| 特性         | 站立（U）                | 趴下（Space）            | RL 行走（WASD） |
| ------------ | ------------------------ | ------------------------ | --------------- |
| **控制方式** | PD 控制                  | PD 控制                  | ONNX LSTM 策略  |
| **Kp**       | 20.0（渐进）             | 20.0（渐进）             | 20.0            |
| **Kd**       | 0.7（渐进）              | 0.7（渐进）              | 0.7             |
| **目标位置** | `*_stand_pos`            | `*_liedown_pos`          | 策略网络输出    |
| **FSM 状态** | STANDUP                  | STANDUP                  | RL_MIX          |
| **子状态**   | RecoveryStand::standUp() | RecoveryStand::lieDown() | -               |
| **跳转条件** | body_height 达标         | body_height 足够低       | -               |

**核心结论**：站立和趴下都是 **纯 PD 关节位置插值控制**，不经过 ONNX 模型。ONNX 策略只在 `RL_MIX` 状态（机器人已经站稳后）才介入。`Motion_State_RecoveryStand` 是处理所有起坐/趴下/收腿/翻身的核心子状态机。





我先确认两个细节：训练时的速度指令范围（决定摇杆缩放），以及当前是否有虚拟手柄设备：


虚拟手柄设备已在 `/dev/input/js0`。再查训练时的速度指令范围：


再确认 ONNXInference 接口、main.cpp 和 CMakeLists：


**第三步：MainController 集成手柄输入与 FSM 跳转逻辑**：


现在修改 main_controller.cpp：


现在在文件末尾添加两个新方法的实现：


**第四步：CMakeLists 更新 + 编译验证**：


编译通过。现在验证 GamepadReader 能否正确发现设备——先确认虚拟 F710 对应的 event 设备号：


编译通过。写一个临时测试程序验证 GamepadReader 运行时行为（设备发现 + 事件读取）：


现在同时运行注入脚本和冒烟测试（需 sudo，uinput 需要 root）：


发现问题：存在残留的旧设备 event17，而新注入的事件走的是新设备。修正为**同时监听所有 F710 设备**（多路 select），避免选错残留设备：


重构为多设备监听。先改头文件：


重写 .cpp 为多设备版本：


重新编译并再次运行闭环测试：


还有个隐患：组合键的完整按下/释放序列可能在**同一次 drain 批处理**中完成，批处理后只看到最终状态。需要改为**每个按键事件后**都检测一次：


LIEDOWN 成功但 STANDUP 和 W 仍丢失。我找到了根因：**每 2 秒的重新扫描会无条件重置按键/摇杆状态**，导致事件在下一次轮询前被清零。修复为仅当打开了**新设备**时才重置状态：


重新编译并验证：


重新编译测试并运行（修复了打印覆盖问题）：


当前正在调试最后一个问题。给你完整状态：

**进展**：运控代码全部写完并编译通过，闭环测试中：
- ✅ W/Q 摇杆速度（ly=-1、rx=-1）正确收到
- ✅ Space（RB+LB）趴下请求正确触发
- ❌ **U（LB+Y）站立请求始终没触发** —— 已排除是打印覆盖的显示问题，是真没检测到

代码逻辑上站立/趴下用的是同一套闩锁检测，但站立总是丢。我现在加调试输出定位 LB+Y 事件到底有没有到达、状态机哪一步出错：


调试输出暴露了关键矛盾：`LB released: saw_y=0`，但 Y↓ 事件明明先到达了。加更详细的状态打印：


输出非常诡异：`KEY code=0x134 val=1` 刚写入，下一行 `combo` 里 `y=0`。这不合常理，先完整读一遍当前代码检查是否有错：


重新编译主库和测试，再次验证：


全部验证通过！STANDUP、LIEDOWN、W/Q 摇杆全部正确检测。清理临时测试文件并记录这个坑：


全部完成了。总结一下这次的工作和遇到的问题：

## 最终实现

按你的要求，**没有做键盘模块**，运控完全对接 `keyboard_control.py` 的虚拟 F710 手柄：

| 文件                                                         | 改动                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [gamepad_reader.h/.cpp](file:///home/qiyuan/Softwares/qiyuan_mc/src/input/gamepad_reader.cpp)（新增） | 通过 evdev 读取 `/dev/input/eventX` 的虚拟 F710；多设备并行监听 + 2s 重扫描；LB+Y/RB+LB 闩锁式组合键检测；摇杆归一化（含死区） |
| [control_fsm.h/.cpp](file:///home/qiyuan/Softwares/qiyuan_mc/src/fsm/control_fsm.cpp) | 站立/趴下纯 PD 插值（捕获起始姿态修复了原来插值不收敛的 bug）、增益 1s 渐进、settle 稳定段、新增 FOLDLEGS 趴下状态 |
| [main_controller.h/.cpp](file:///home/qiyuan/Softwares/qiyuan_mc/src/main_controller.cpp) | 移除自动站立；U 键 PASSIVE→STANDUP→RL_MIX，Space 键 RL_MIX→FOLDLEGS→PASSIVE；进 RL 前 resetLSTM + 清零 action；仅 RL_MIX 跑 ONNX；摇杆满量程映射 ±1.0（与训练范围一致） |
| CMakeLists.txt                                               | 加入新源文件                                                 |

## 调试中解决的 3 个问题

1. **残留设备干扰**：上次会话遗留的死 F710 设备还在系统里 → 改为监听所有 F710 设备
2. **重扫描清空状态**：2s 重扫描会无条件重置按键状态，把还没被消费的请求清掉 → 仅当打开新设备时才重置
3. **最隐蔽的一个**：站立组合键始终检测不到 —— 这台机器的内核头文件里 `BTN_NORTH=0x133`，而 keyboard_control.py（python-evdev）发的是 `0x134`，两者正好相反。已改为硬编码 `GP_Y=0x134`，这个坑已记入记忆

## 闭环验证结果（模拟注入测试）

```
[EVENT] >>> STANDUP request   ✅ U 站立
[STICK] ly=-1                 ✅ W 前进
[STICK] rx=-1                 ✅ Q 左转
[EVENT] >>> LIEDOWN request   ✅ Space 趴下
```

编译通过，可以直接联调了：先起 MuJoCo 仿真 + `keyboard_control.py`（sudo），再跑 `qiyuan_mc/build/mc_ctrl`，按 U 站立后用 WASD/QE 控制。
