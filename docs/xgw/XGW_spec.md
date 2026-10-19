# XGW（xgb 轮足）接口规格 —— 基于实测取证

> 本文件所有数值均来自**同刻实测**（非推断、非文档）。取证方法见第 9 节。
> 取证对象：Matrix `xg_wheel` 模型集在 MuJoCo + UE 联合仿真下运行 `RL_MIX` 时的真实数据通路。
> 取证时间：2026-10-19（第二轮，含手柄全向行走/转向）；日志：
> `/tmp/xgw_cap/wire4.log`（eCAL 原始报文，182 MB，约 175 s）、`/tmp/xgw_cap/dense_obs.txt`（hook 抓的 obs，78 MB，15406 帧 ×2 实例）。
> 分析脚本：`/tmp/xgw_cap/v8.py`（镜像/regime/精确常数）、`v9.py`（tau_ff/静止语义）、`v10.py`（gyro/头部/共享/last_act）。

---

## 1. 机型与关节定义

| 项 | 值 | 依据 |
|---|---|---|
| 机型 | xgb 四足轮腿（每条腿 ABAD/HIP/KNEE + 轮 FOOT） | `xgw/xg_wheel.urdf`、`xgw.xml`（FOOT 无限位、独立惯量） |
| 关节总数 | **16**（4 腿 × 4 关节） | leg_data 每腿 4 组 `q/qd` 均非零 |
| 腿序 | `[FR, FL, RR, RL]`（索引 0,1,2,3） | 动作槽 4 组与 `qd_des_WHEEL[0..3]` 一一对应 |
| 腿内关节序 | `[ABAD, HIP, KNEE, FOOT(轮)]` | 4 槽结构 + 第 4 槽与轮子速度指令 corr=1.0000 |

**q_default（关节零位，实测）**

```
q_default_ABAD = [+0.1, -0.1, +0.1, -0.1]   # 即 0.1 * [ +1,-1,+1,-1 ]
q_default_HIP  = +0.8    (四腿相同)
q_default_KNEE = -1.5    (四腿相同)
q_default_FOOT =  无（连续旋转，无界；实测可达 385 rad）
```

依据：`obs[12:24] = q_meas - q_default` 的回归截距恰为 `(-0.1, -0.8, +1.5)`（腿 0/2）/`(+0.1, -0.8, +1.5)`（腿 1/3），slope=1.0、corr=1.0000。
注意：**实测的 `q_meas` 不带 per-leg 符号翻转**（站立时 `q_hip` 四腿同为 +0.70~+0.85，`q_knee` 同为 -1.54）；`q_default_ABAD` 的 ±0.1 符号与实测 abad 的符号模式一致。

---

## 2. 模型 IO

| 模型 | 输入 | 输出 | 说明 |
|---|---|---|---|
| policy（walk / gait_walk / balanceStand / handstand / crawl / backflip / jump） | `[1, 56]` | `[1, 16]` | 16 维动作，无 clip（训练时 `clip_actions=None`） |
| odom | `[1, 33]` | `[1, 3]` | 输出 3 维 |

交叉验证：hook 内 XG 的键为 odom `1x29` / 策略 `1x48`；XGW 实测为 **`1x33` / `1x56`**（`dense_obs.txt` 行首 `n=33`、`|extra n=56`）。
差值 `33-29 = 4`、`56-48 = 8 = 4(jvel) + 4(last_act)` —— 与"每腿多 1 个轮子、共 16 关节"的结构**完全自洽**。

---

## 3. policy obs（56 维）布局

| 索引 | 内容 | 生成方式 | 实测证据（第二轮，n=15406） |
|---|---|---|---|
| `[0:3]` | `2 x base_vel (vx,vy,vz)` | 仿真未提供 → **实测全程恒为 0** | `max|.| = 0.000000`；镜像不翻转 |
| `[3:6]` | `gyro x 0.25` | 原始 IMU 陀螺 | **`= 0.25 x sdk_robotstate.f5`**：slope `0.25000/0.24998/0.25000`、corr `0.99999/0.99996/1.00000`、ic ≤2e-6 |
| `[6:9]` | `roll, pitch, 0` | 四元数解算（`mujoco_state.f13`），**原值不缩放** | slope `0.99929/0.99994`、corr `0.99956/0.99993`；`[8]` `max|.|=0.000000` |
| `[9:12]` | `2 x vel_cmd (vx,vy,wz)` | 手柄指令 | 实测 max `2.0575 / 2.0000 / 0.5000` ⇒ ×2（真实指令上限 1.03/1.0/0.25） |
| `[12:24]` | `jpd[12] = q_meas - q_default`，**仅 12 个腿关节**（不含轮），**不缩放** | | slope `0.9994~1.0000`、corr ≥0.9997（leg-major 索引：`idx = leg*3 + joint`） |
| `[24:40]` | `jvel[16] = qd_meas x 0.05`，**含全部 16 关节（含轮）** | | slope `0.04962~0.05000`、corr ≥0.9955（leg-major：`idx = leg*4 + joint`） |
| `[40:56]` | `last_act[16]` = **上一拍的合并动作**（严格 1 拍延迟，无 slew/限幅） | 见第 5.4/6 节 | 见下 |

**`last_act` 语义（第二轮定论，`v10.py` 第 4 节）**

把 hook 抓的 A/B 合并动作按时间轴平移后与 `obs[40:56]` 逐槽回归：

| 平移量 | 16 槽 slope 中位 | slope 范围 | min\|corr\| | max\|ic\| |
|---|---|---|---|---|
| **-1（上一拍）** | **1.00000** | **1.00000 ~ 1.00000** | **1.00000** | **0.000000** |
| 0（当拍） | 0.98025 | 0.95824 ~ 0.99767 | 0.95824 | 0.024080 |
| -2 | 0.98025 | 0.95826 ~ 0.99792 | 0.95825 | 0.024205 |
| +1 | 0.92828 | 0.86448 ~ 0.99206 | 0.86449 | 0.084670 |

⇒ `last_act` **精确等于上一拍 `0.5*(A_raw + unmirror(B_raw))`**，不做任何附加处理。

**顺序注意**：`jpd` 只含 12 个腿关节（轮被排除），而 `jvel` 含 16 个（轮被包含）。这是 56 与 33 两个维度能对齐的关键。

---

## 4. odom 输入（33 维）布局

| 索引 | 内容 |
|---|---|
| `[0:2]` | `roll, pitch` |
| `[2:14]` | `jpd[12]`（同 policy `[12:24]`） |
| `[14:30]` | `jvel[16]`（同 policy `[24:40]`） |
| `[30:33]` | `gyro`（同 policy `[3:6]`） |

**共享性为精确零差（非近似）**，实测 `max|diff| = 0.000e+00`（n=15406，四个切片全部逐元素相同）：

```
odom[0:2]   == policy[6:8]
odom[2:14]  == policy[12:24]
odom[14:30] == policy[24:40]
odom[30:33] == policy[3:6]
```

⇒ odom 与 policy **复用同一批数组**（一份关节/姿态数据喂两个模型）。实现时应先算一次，再分别拼装，避免二次计算引入不一致。

---

## 5. A/B 双实例镜像映射（相关系数 ±1.0000，精确解）

两个策略实例 A（正常）/ B（镜像），输出合并 `0.5*(A + unmirror(B))`。B 的输入按下表由 A 的输入映射而来（`v8.py`，索引级求解，`min|corr| = 1.0000`）。

### 5.1 头部
```
gyro[3:6] : x 翻符号, y 不翻, z 翻符号        (corr -1.0000 / +1.0000 / -1.0000)
[6:9]     : roll 翻符号, pitch 不翻, [8] 恒 0 (corr -1.0000 / +1.0000)
[9:12] vcmd: vcmd_x 不翻, vcmd_y 翻, vcmd_z 翻 (corr +1.0000 / -1.0000 / -1.0000)
[0:3] bv    : 全 0，无法判定（按不翻转处理）
```

### 5.2 jpd[12]（每腿 3 槽，仅 ABAD 翻符号）
```
MIR  = { 3, 4, 5,  0, 1, 2,  9,10,11,  6, 7, 8 }
SIGN = { -1, 1, 1 } 重复四次
```

### 5.3 jvel[16] 与 last_act[16]（每腿 4 槽，仅 ABAD 翻符号）
```
MIR  = { 4, 5, 6, 7,  0, 1, 2, 3,  12,13,14,15,  8, 9, 10, 11 }
SIGN = { -1, 1, 1, 1 } 重复四次
```
即腿 0↔1、腿 2↔3 对换；组内仅第 1 槽（ABAD）翻符号。
（与已验证的 XG `MIR/MIR_SIGN` 约定一致，只是由 12 泛化为 16。）

**物理含义自洽性**：上述映射等价于矢状面反射（x-z 平面，左右为 y 轴）下的轴矢量变换 `(ωx,ωy,ωz) → (-ωx,+ωy,-ωz)`，与 gyro x翻/y不翻/z翻、roll翻/pitch不翻、vcmd_y/vcmd_z翻 **完全一致**。

### 5.4 合并后动作的回填
`last_act`（obs[40:56]）记录的是**上一拍合并后的动作**，即：
```
merged = 0.5*(actA + unmirror(actB))         # unmirror 用 5.3 的 MIR/SIGN
last_act[next_tick] = merged[this_tick]
```
可直接回填，无需逆运动学、无需逆缩放。

---

## 6. 控制律（RL_MIX 状态，实测）

FSM 在 RL_MIX 内有**两套 regime**，由是否有速度指令切换（`v8.py`）：

| | **行走 regime** | **静止保持 regime** |
|---|---|---|
| 触发条件 | `|vcmd| != 0` | `|vcmd| == 0`（实测 max 0.0000，n=6339） |
| 腿 kp / kd | **20.0 / 0.7** | **130.0 / 2.0** |
| 轮 kp / kd | kp_foot = **0**，kd_foot = 0.1 | kp_foot = **20**，kd_foot = 0.1 |
| 腿位置指令 | `q_des_leg = q_default + 0.25 * act_leg` | `q_des_leg ≈` 默认直立姿态（hip 中位 0.7831 / meas 0.7848，knee 中位 -1.4893 / meas -1.5359，abad 中位 ≈0，±0.1） |
| 腿速度指令 | `qd_des_leg = 0` | `qd_des_leg = 0` |
| 腿 tau_ff | 0 | 0 |
| 轮位置指令 | `q_des_foot ≡ 0`（实测） | `q_des_foot` = 进入该 regime 时的实测轮角，**整段保持**（见下） |
| 轮速度指令 | `qd_des_foot = 11.25 * act_wheel` | 仍有 `qd_des_foot`，但实测轮速≈0 |

> **重要修正**：上一版规格把 `kp130/kd2` 当作 RL 行走增益，**错误**。130/2 只出现在**无速度指令的静止保持段**；行走时腿增益是 **20/0.7**（与 XG 四足行走 regime 同值，仅 kd 从 1.0 变 0.7）。

### 6.1 腿关节（ABAD/HIP/KNEE）
```
q_des_leg  = q_default + 0.25 * act_leg     # 行走 regime
qd_des_leg = 0
kp_leg = 20.0    kd_leg = 0.7              # 行走；静止为 130.0 / 2.0
tau_ff = 0
tau = kp*(q_des - q) + kd*(qd_des - qd)
```

### 6.2 轮子（FOOT）
行走 regime 实测（`v9.py`，n=36264）：
```
q_des_foot  = 0                              # 实测恒为 0
qd_des_foot = 11.2486780 * act_wheel         # 速度指令（ic 8.8e-4, corr 0.99982）
kp_foot     = 0     kd_foot = 0.1
tau_ff_foot = qd_des_foot / 90 = 0.1249853 * act_wheel
              # vs qd_des_foot: slope 0.011111 (=1/90), ic 0, corr 1.0000
              # vs act_wheel  : slope 0.1249853, ic 1e-5, corr 0.99982
              # 值域 [-0.4906 .. 0.3794]
```
即轮子**纯速度项 + 一个 qd_des/90 的前馈**，位置项被置 0。
注意：`gate.proto` 中 `q_des_foot` 的旧注释"仿真器恒 0"**已过期**——静止 regime 下实测非 0（可达 385 rad）。

### 6.3 静止保持 regime 的 `q_des_foot`（观察，未完全定死）
进入 `kp130` 段时，`q_des_foot` 被**一次性置为当时的实测轮角**并**整段保持不变**（相邻报文间 `d(q_des_foot) == 0`，corr 0.0000，**不是逐拍积分**）：
```
sec22-30 : q_des_foot ≡ 4.1528     (meas 4.18~4.20)
sec49-54 : q_des_foot ≡ 183.7412   (meas 183.77)
sec143-172: q_des_foot ≡ 42.2549
```
进入 `kp20` 段时归 0。
**未定死**：该 latch 的更新时机（每个 100 Hz 拍？状态进入时？）——跨段混算会得到 `median|q_des_foot-q_meas_foot| = 20.96`，必须**按段判断**。影响评估：**不阻塞实现**（自实现时按 `q_des_foot := q_meas_foot` 每拍刷新即可，与 kp130 的高增益组合效果等价）。

### 6.4 动作缩放约定（回填 `last_act` 时用）
```
act_leg   = (q_des_leg - q_default) / 0.25
act_wheel = qd_des_wheel / 11.2486780
```

**行走 regime 精确常数（`v8.py` / `v10.py`，n=36264，用上一拍合并动作回归）**

| 槽 | slope | ic | corr |
|---|---|---|---|
| abad | 0.2498839 | +1.1e-5 | 0.9997074 |
| hip | 0.2498369 | +4.1e-5 | 0.9994006 |
| knee | 0.2499198 | +6.9e-5 | 0.9996257 |
| wheel（→`qd_des_foot`） | **11.2486780** | +8.8e-4 | 0.9998229 |
| wheel（→`tau_ff_foot`） | **0.1249853** | +9.8e-6 | 0.9998229 |

---

## 7. 未决项与风险（不猜测，明确标注）

1. **`last_act[40:56]` 的前 3 槽** —— **已闭合**（第二轮）：精确等于上一拍合并动作，16 槽 slope 1.00000 / corr 1.00000 / ic 0。第一轮观测到的"仅 corr 0.79~0.86"是**实例配对错误 + 未做 1 拍时移**所致。
2. **gyro 系数** —— **已闭合**（第二轮）：`= 0.25 x sdk_robotstate.f5`，三轴 slope 0.25000/0.24998/0.25000、corr ≥0.99996。第一轮的 0.223/0.230 系参考量（`mujoco_state.f14`，经滤波的真值）与 obs 非同源、且存在噪声导致的斜率低估。
3. **静止 regime 的 `q_des_foot` latch 时机** —— 见 6.3，未定死，不阻塞实现。
4. **`[0:3]` base_vel 恒 0**：说明当前仿真链路未提供机身线速度。真机或后续版本若提供，需确认是否同样 ×2。
5. **`[9:12]` 的 ×2 倍率**：由 obs 实测上限（2.06/2.00/0.50）反推，与 Matrix 配置 `cmpc_x_vel=1.0` 自洽；倍率本身未用独立真值直接证明。
6. **其它 skill 的 57/59 维输入**：本规格覆盖 RL_MIX（56）；`gait_walk / handstand / crawl / backflip / jump / balanceStand` 的输入维数未逐一实测，多出的 1~2 维归属未定。

---

## 8. 与 XG（四足）对照

| | XG | XGW |
|---|---|---|
| 每腿关节 | 3 | **4**（+FOOT 轮） |
| 关节总数 | 12 | **16** |
| odom 输入 | 29 = 2+12+12+3 | **33** = 2+12+16+3 |
| policy obs | 48 = 12+12+12+12 | **56** = 12+12+16+16 |
| 动作 | 12 | **16** |
| jpd | q-q_default（12，不含轮） | 同（12，不含轮） |
| jvel | qd×0.05（12） | **qd×0.05（16，含轮）** |
| 轮子 | 无 | **速度模式**：`qd_des=11.2487×act`，`tau_ff=qd_des/90`，kp_foot=0 kd_foot=0.1（行走）/ kp20（静止） |
| 腿行走增益 | kp20 / kd1.0 | **kp20 / kd0.7** |

---

## 9. 取证方法与复现

**取样原理**：让观测点与真值点在**同一时钟基准**下记录，再用直接回归求缩放。

1. **hook 改造**：`/home/qiyuan/Softwares/qiyuan_mc/tools/onnx_hook/onnx_dump_hook.cpp`
   - `nowMs()` 增加 `HOOK_EPOCH_TS=1` 开关 → 返回 `CLOCK_REALTIME` 绝对微秒（默认行为不变）。
   - 新增 `samplePeriod()` 读 `HOOK_SAMPLE`（默认 50，`=1` 全速），并修复 `period==1` 时 `count % period == 1` 永不成立的取样 bug。
   - `HookedRun` 泛化到 XGW，使 `1x33`/`1x56` 也落盘（策略原始输出 16 维）。
   - 编译：`g++ -O2 -std=c++17 -shared -fPIC -I/usr/local/include onnx_dump_hook.cpp -o libonnx_dump_hook_epoch.so -ldl -lpthread`
     （原 `.so` 备份为 `libonnx_dump_hook.so.orig_20261019`）
2. **eCAL 订阅器**：`/tmp/xgw_cap/sub3.cpp`（纯 raw hex dump，不依赖 proto；白名单 4 topic：`leg_data`/`leg_cmd`/`sdk_robotstate`/`mujoco_state`），编译为 `/tmp/xgw_cap/xgw_sub3`。
3. **运行**（务使订阅器先于仿真启动）：

```bash
# 终端 1
/tmp/xgw_cap/xgw_sub3 /tmp/xgw_cap/wire4.log 1 900      # decim=1, 900s 窗口
# 终端 2
cd /home/qiyuan/Softwares/Matrix
export LD_PRELOAD=/home/qiyuan/Softwares/qiyuan_mc/tools/onnx_hook/libonnx_dump_hook_epoch.so
export HOOK_EPOCH_TS=1
export HOOK_SAMPLE=1 HOOK_POL_FULL=1
export HOOK_OBS_LOG=/tmp/xgw_cap/dense_obs.txt
bash scripts/run_sim.sh 2 0 0 0 1
```
参数含义（位置参数）：`arg1=2`(机器人 xgw) `arg2=0`(场景 custom 开阔平地，1=wh) `arg3=0` `arg4=0` `arg5=1`(启用 MuJoCo)。
注意：`run_sim.sh` 会用 `sed` **覆写** `simulate/config.yaml` 的 `robot_scene`，故场景必须用 arg2 传，手改该文件无效。
本轮用户操作：前后左右行走 + 左右旋转（手柄全向测速）。

4. **报文解码要点（字段号 + packed 编码）**
   - **proto3 `repeated float` 默认 packed**（wire type = 2，length-delimited 内为定长 float 串）。
     **踩坑**：若按旧式"wt=2 即跳过"解析，则所有字段全空、回归 corr=0。必须仅在 `len % 4 == 0` 时按 `<f` 解成 float 序列。
   - 字段表（本轮实测，wire4.log）：

   | topic | 字段号 | 含义 |
   |---|---|---|
   | `leg_data` | 1..4 `q_abad/hip/knee/foot`、5..8 `qd_*`、9..12 `τ_fb_*` | 仅 12 个字段（**不含** quat/gyro/acc） |
   | `leg_cmd` | 1..4 `q_des_*`、5..8 `qd_des_*`、9..12 `kp_*`、13..16 `kd_*`、17..20 `τ_ff_*` | 20 个字段 |
   | `mujoco_state` | 1..4 `q_*`、5..8 `qd_*`、9..12 `τ_*`、13 `quat[w,x,y,z]`、14 `gyro[3]`、15 `acc[3]`、19 `position[3]`、20 `v_world[3]` | 仿真真值 |
   | `sdk_robotstate` | 5 `gyro[3]`、6 `acc[3]`、7/8/9/10 其它、13 标量(时间戳) | IMU 原始 |

   ⚠️ 旧版规格把 quat/gyro/acc 记在 `leg_data` 下，**错误**：本轮实测 `leg_data` 只有 1..12，姿态/IMU 在 `mujoco_state` 与 `sdk_robotstate`。

---

## 10. 实施清单（待办）

- [ ] 模型层：`common_types.h` 的 `NUM_JOINTS_PER_LEG 3→4`；16 维指令/状态结构；轮子无界处理
- [ ] 参数层：xgw 的 motor_config / robot 参数（q_default = abad±0.1 / hip0.8 / knee-1.5；轮无界）
- [ ] obs 构建：严格按第 3 节（jpd 取 12、jvel 取 16、last_act = 上一拍合并动作）
- [ ] odom 构建：按第 4 节，**与 policy obs 共享同一数组**（一次计算）
- [ ] A/B 镜像：按第 5 节 MIR/SIGN（jpd 与 jvel/last_act 两套）
- [ ] FSM 控制（两 regime，见第 6 节）：
      - 行走：腿 `q=q_def+0.25·act, kp20/kd0.7`；轮 `q_des=0, qd_des=11.2487·act, tau_ff=qd_des/90, kp0/kd0.1`
      - 静止：腿 `kp130/kd2` 保持姿态；轮 `q_des:=q_meas, kp20/kd0.1`
- [ ] LSTM 复位：切状态时复位 policy/odom 的 LSTM
