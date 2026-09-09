# UniLab 与 mjlab MicroDuck 对比（2026-09-08）

## 结论

在本机的 `microduck_velocity_flat` PPO 对比中，UniLab 没有显示出可以支持
迁移主训练链路的效率或早期学习质量优势：

- 100 iterations 时，UniLab CPU MuJoCo 与 mjlab mjwarp 的训练阶段吞吐基本相同。
- UniLab 总墙钟时间慢 `8.0%`，峰值 RSS 约为 mjlab 的 `4.8x`。
- 同样 491.52 万 env steps 后，UniLab 的终段平均 reward 低 `40.9%`，
  episode length 低 `42.7%`，跌倒终止更多。
- UniLab 的 MicroDuck `mjwarp` host profile 在 2048 envs 仅约 `1.35k steps/s`，
  而当前 mjlab 约 `25.16k steps/s`，不适合当前大 batch 训练。

这是单 seed、100 iterations 的早期学习对比，不是最终收敛或真机精度结论。

## 环境

| 项目 | 值 |
| --- | --- |
| CPU | Intel Core Ultra 9 285K，24 logical CPUs |
| GPU | NVIDIA GeForce RTX 5060，8151 MiB |
| 当前项目 | `microduck_rl` `2b25a48`，mjlab 1.3.0，Warp 1.12.0 |
| UniLab 分支 | `8530dbc`，UniLab 1.0.0，Warp 1.17.0 |
| task | velocity flat |
| actor / critic obs | 61 / 76 |
| action | 14 |
| physics / control | 200 / 50 Hz |
| PPO rollout | 24 steps/env |
| seed | 42 |

UniLab 自带 alignment audit 结果为 `184 MATCH / 0 GAP / 6 NOTE`。两边 actor、
critic、PPO 主要参数、BAM action、observation 和 reward 契约对齐。
UniLab 默认配置额外打开 mirror loss，100-iteration 可比实验通过
`algo.algorithm.symmetry_cfg=null` 将其关闭，与当前 mjlab 的 `ENABLE_SYMMETRY=False`
保持一致。

## 效率结果

### 64 envs，5 iterations（warm cache）

| 路径 | 外部墙钟 | 末轮日志 FPS | 峰值 RSS |
| --- | ---: | ---: | ---: |
| mjlab mjwarp | 19.81 s | 1,250 | 3.18 GB |
| UniLab mjwarp | 18.48 s | 920 | 3.20 GB |

64 envs 时总时间被启动成本主导，不用该墙钟差作速度结论。

### 2048 envs，5 iterations（warm cache）

| 路径 | 外部墙钟 | 训练吞吐 | 峰值 RSS |
| --- | ---: | ---: | ---: |
| mjlab mjwarp | 31.28 s | 约 23.7k steps/s | 3.17 GB |
| UniLab CPU MuJoCo | 54.10 s | 18.68k steps/s（整段） | 15.24 GB |
| UniLab mjwarp host profile | 195.06 s | 1.35k steps/s | 3.23 GB |

### 2048 envs，100 iterations（4,915,200 env steps）

| 路径 | 训练阶段吞吐 | 外部墙钟 | 峰值 RSS |
| --- | ---: | ---: | ---: |
| mjlab mjwarp | 25.16k steps/s | 215.49 s | 3.14 GB |
| UniLab CPU MuJoCo | 25.47k steps/s | 232.82 s | 14.95 GB |

UniLab 训练阶段吞吐高 `1.3%`，在本次短测的波动范围内；包含启动、
CPU worker 和 backend 准备后，总墙钟高 `8.0%`。

## 早期学习质量

指标由仓库自带的 `scripts/microduck_alignment_compare.py` 计算；终段是最后
20 iterations 的均值。

| 指标 | UniLab CPU MuJoCo | mjlab mjwarp | 差异 |
| --- | ---: | ---: | ---: |
| 终段 mean reward | 9.600 | 16.230 | -40.9% |
| 终段 episode length | 125.205 | 218.441 | -42.7% |
| 达到各自终段 reward 80% | iter 82 | iter 83 | 接近 |
| 终段 tilt/fell-over 计数 | 14.529 | 5.398 | UniLab +169.2% |
| 最后一轮 mean reward | 13.092 | 21.260 | -38.4% |
| 最后一轮 episode length | 166.57 | 280.43 | -40.6% |

reward term 的差异不是单纯缩放：UniLab 的 linear velocity、upright、pose 和
head tracking 等正奖励同时较低，表明该时点的策略确实存活更短，而不只是
日志命名差异。

## 复现命令

UniLab（为了与当前 mjlab 对齐，关闭默认 mirror loss）：

```bash
uv run microduck-train --algo ppo --task microduck_velocity_flat --sim mujoco \
  algo.num_envs=2048 algo.max_iterations=100 algo.seed=42 +env.seed=42 \
  algo.algorithm.symmetry_cfg=null training.no_play=true \
  training.play_env_num=4 training.logger=tensorboard
```

mjlab（在原 `microduck_rl` checkout 中）：

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.max_iterations 100
```

生成 reward 对比：

```bash
uv run scripts/microduck_alignment_compare.py \
  --unilab <unilab-run-dir> \
  --upstream <mjlab-run-dir> \
  --output comparison.md --json comparison.json
```

## 限制和下一道门禁

- 只测了一个 seed；无法得到方差或统计显著性。
- 100 iterations 还处于学习早期，不能代表最终 gait 或 sim2real 质量。
- UniLab 用 CPU MuJoCo，mjlab 用 MuJoCo Warp；即使契约一致，后端数值和接触
  并不保证逐步相同。
- UniLab 使用 mujoco-warp 3.10.0.3，原项目是 3.8.1。
- 现有 UniLab 分支只覆盖 5 个 task，没有 rough、roller、backlash、ball kick、
  spin 和 roulade 等当前主项目任务。
- 迁移决策前至少需要 3 seeds × 2000 iterations，比较相同命令电池、
  rollout 成功率、稳态速度误差、跌倒率和 ONNX CPU MuJoCo rehearsal。

## 验证

- `pytest tests/ -q`: `99 passed`, 2 个已知 backend-option warnings。
- alignment audit: `184 MATCH / 0 GAP / 6 NOTE / 0 status mismatch`。
- actor / critic / action shape: `61 / 76 / 14`。
- 所有 smoke 和 100-iteration run 的 `nan_state` 都是 0。
