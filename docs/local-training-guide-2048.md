# Microduck 本机训练指南（RTX 5060 · 2048 envs · 可续训）

> 本文档针对**当前这台机器**编写（2026-09-08 实测）：
> - 显卡：NVIDIA RTX 5060，**8 GB 显存**（约 2.4–2.9 GB 被桌面显示占用）
> - 环境已配置好：`uv sync` 完成、wandb 已登录（`weilan157`，项目 `mjlab_microduck`）
> - 实测结论：**2048 envs 安全**（峰值 ~5.1 GB），约 **24k steps/s、~2.1 s/轮**，
>   即 **~35 分钟 / 1000 轮**。
> - 冒烟测试用 64 envs × 5 轮，正式训练用 2048 envs。
>
> 大训练（4096 envs、多卡）仍建议走云端 `--hf-jobs`，不占本机。

---

## 0. 总览（先看这条）

```bash
# 首次启动（例子：跑 2000 轮一个里程碑）
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 2048 \
    --agent.max_iterations 2000 \
    --agent.run-name gait01

# 中断后从断点继续（每次加你想「多跑」的轮数）
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 2048 \
    --agent.max_iterations 2000 \
    --agent.run-name gait01-cont \
    --agent.load-checkpoint model_2000.pt \
    --agent.resume True
```

**核心规则（务必理解，否则续训会跑错轮数）：**
`--agent.max_iterations N` 的含义是「**跑到当前轮数 + N**」：

- 新跑：从 0 跑到 N；
- 续训：从断点（如 2000）继续跑到 **2000 + N**（不是 N，也不是“跑到 N”）。

所以轮数不确定时，把它当成「**这一轮会话想多跑多少**」来传，而不是最终目标。

---

## 1. 第一步永远是冒烟测试

改动任务/配置后、正式长跑前，先跑 5 轮确认不炸（约 7 秒，无需联网 wandb）：

```bash
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5
```

通过标准：能跑完、`nan_state` 全程 = 0、所有罚项 ≤ 0、obs 是 61D。

---

## 2. 选任务

```bash
uv run list-envs          # 全部任务
```

常用 MicroDuck 任务（Flat=平底 / Rough=崎岖地形，插 `-Backlash-` 表示带齿轮间隙仿真）：

| 任务 ID（示例） | 说明 |
|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | **主任务**：走路 + 速度指令 + 头部姿态 |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | 走路 + 摔倒恢复 |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | 从趴/仰/坐站起来并保持 |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | 受控坐↔站 |
| `Mjlab-Roulade-Flat-MicroDuck` | 前滚翻 |
| `Mjlab-Spin-Flat-MicroDuck` | 轮滑原地转圈 |

下文以 `Mjlab-Velocity-Flat-MicroDuck` 为例，换成任何任务 ID 即可（`<TASK_ID>` 通指）。

---

## 3. 正式启动训练

```bash
uv run train <TASK_ID> \
    --env.scene.num-envs 2048 \
    --agent.run-name <给你的run起个名> \
    --agent.max_iterations <本会话轮数>
```

参数说明：

| 参数 | 作用 |
|---|---|
| `--env.scene.num-envs 2048` | 并行环境数。**本机就固定用 2048**（4096 峰值 7.1 GB 余量太险） |
| `--agent.run-name xxx` | 本地目录与 wandb run 的名字后缀，方便辨认/续训定位 |
| `--agent.max_iterations N` | **本会话新增轮数**（见第 0 节语义）。不传则用任务默认（velocity = 50000） |
| `--video True` | （可选）训练中录视频，会多吃显存，8GB 下谨慎 |
| `--env.seed` / `--agent.seed` | 复现种子，默认 42 |

**示例（首跑 2000 轮）：**

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 2048 \
    --agent.run-name gait01 \
    --agent.max_iterations 2000
```

启动后看终端结尾打印的 wandb 链接，实时曲线都在上面：
`https://wandb.ai/weilan157-null/mjlab_microduck/runs/<run_id>`

### 训练轮数怎么定（你说了“不确定”）

- **简单技巧类**（Roulade、Spin、StandUp 等）：先按 **1000–2000 轮** 一个里程碑试。
- **gait / 课程重的**（Velocity、VelStand、Rough 系列）：先按 **2000–3000 轮** 一个里程碑试，
  通常需要几段续训累计 **几千到上万轮**（AGENTS.md 参考预算在 4096 envs 是 4000–6000 轮；
  本机 2048 envs 每轮经验减半、但课程按轮推进，实际以曲线为准）。
- **判断“够不够 / 要不要继续”** 见第 5 节——别靠猜，靠曲线和实测。

---

## 4. 中断后如何续训

中途想停：直接 `Ctrl+C` 即可。**每 250 轮会自动存一个 checkpoint**
（`logs/rsl_rl/<experiment>/<run目录>/model_<轮数>.pt`，velocity 的 `save_interval=250`），
中断最多损失不到 250 轮的进度。

### 续训命令

```bash
uv run train <TASK_ID> \
    --env.scene.num-envs 2048 \
    --agent.run-name <同一语义的新名字，如 gait01-cont> \
    --agent.max_iterations <想多跑的轮数> \
    --agent.load-checkpoint model_<断点轮数>.pt \
    --agent.resume True
```

**示例：gait01 跑完 2000 轮后，想再跑 2000 轮（到 4000）：**

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 2048 \
    --agent.run-name gait01-cont \
    --agent.max_iterations 2000 \
    --agent.load-checkpoint model_2000.pt \
    --agent.resume True
```

会从 `model_2000.pt` 载入权重/优化器/观测归一化/当前轮数，接着跑到 **4000**。

### 续训注意事项（踩过坑的都在这）

1. **`max_iterations` 是“新增轮数”**：上面例子若再传 2000，就是从 2000 跑到 4000；
   想多跑就继续加大并再 resume。
2. **checkpoint 文件名要写精确**（如 `model_2000.pt`）。不写的话默认按字母序挑最新
   `model_*.pt`，而 `model_9999.pt` 字母序会排在 `model_10000.pt` 后面（非零填充的数字不按大小排），
   可能载错断点。**永远显式传 `--agent.load-checkpoint`。**
3. **run 目录/任务名靠 `--agent.load-run` 指定**（可选）：默认 `.*` 取**最新一个 run 目录**。
   若你同时有多个任务的 run 或想从较早的 run 续，加上
   `--agent.load-run <logs/rsl_rl/<experiment>/ 下的目录名>`。
4. **每次 resume 都会新建一个本地 run 目录和一个新 wandb run**（名字 = 你传的 `--agent.run-name`），
   它只是把权重/进度接上。想看完整曲线可把多段合起来看。
5. **保持 `--env.scene.num-envs 2048` 不变**（obs/action 维度与任务必须一致；env 数也建议一致）。
6. **seed 保持一致**（默认 42 即可，不必改）。

### 找不到 checkpoint？

本地路径规律：

```bash
ls logs/rsl_rl/velocity/          # 看有哪些 run 目录
ls logs/rsl_rl/velocity/<run目录>/   # 看有哪些 model_*.pt
```

---

## 5. 判断练得好不好 / 该不该停

训练时每轮看 wandb（或终端）这些项（AGENTS.md 硬规则）：

- ✅ **每个 `Episode_Reward/<罚项>` ≤ 0**（如 `action_rate_l2`、`body_ang_vel`）——若有为正，是 bug。
- ✅ `Episode_Termination/nan_state` 恒为 0。
- 📈 **主任务项在涨**（velocity 看 `track_linear_velocity`、`upright`；技巧类看对应 term）。
- 📊 episode length / 摔倒率符合任务预期，别只看总 reward 涨（可能只涨正则项而技巧没学会）。

**停止/继续的决策**：练到曲线平台期后，**用 play 实测**而不是只信曲线：

```bash
# 用 wandb run_id 载入某个 checkpoint 的模型来看效果
uv run play Mjlab-Velocity-Flat-MicroDuck \
    --wandb-run-path weilan157-null/mjlab_microduck/<run_id>
```

感觉不够好 → 回到第 4 节 resume 再跑一段；满意 → 进入第 6 节部署。

---

## 6. 训练完 → 导出 / 部署（简要）

```bash
# 导出 ONNX（烘焙观测归一化，必经路径；用 wandb run id 定位 checkpoint）
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
    --wandb-run-path weilan157-null/mjlab_microduck/<run_id>

# 真机前先在 CPU MuJoCo 彩排（键盘控制）
uv run scripts/infer_policy.py --walking output.onnx
```

---

## 附：本机实测参数速查（velocity 默认值）

| 项 | 值 |
|---|---|
| 每 env 步数 / 轮 | 24 |
| 默认最大轮数 | 50000（建议按里程碑覆盖） |
| 自动保存间隔 | 250 轮 |
| checkpoint 命名 | `model_<轮数>.pt` |
| wandb 项目 | `mjlab_microduck`（entity `weilan157-null`） |
| 本机安全 env 数 | **2048**（峰值 ~5.1 GB / 8 GB） |
| 实测吞吐 | ~24k steps/s，~2.1 s/轮，~35 min/1000 轮 |

> 冒烟/调试用 `WANDB_MODE=offline`（不污染云端项目）；正式训练不要加 offline，
> 否则 run 不会上传、之后 `--wandb-run-path` 找不到。
