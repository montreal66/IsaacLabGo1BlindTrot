# Go1 Blind Trot 实验说明

本分支在 Isaac Lab 的 Go1 rough velocity task 基础上增加了两个盲走（blind locomotion）任务。核心区别是：策略网络和价值网络是否可以看到地形高度扫描（terrain-height scan）。

## 1. Blind：非对称 actor-critic

Task：`Isaac-Velocity-Rough-Unitree-Go1-Blind-v0`

- **actor / policy** 只使用本体感觉（proprioception），不读取地形高度。
- **critic** 保留地形高度作为 privileged observation。
- 因此 height scanner 仍会保留，仅供 critic 使用。
- 增加 terrain-distribution curriculum，并写入 TensorBoard。

这是非对称训练：训练时 critic 有额外地形信息，但部署时 policy 不依赖它。

## 2. Blind Symmetric：对称盲走

Task：`Isaac-Velocity-Rough-Unitree-Go1-Blind-Symmetric-v0`

- actor 和 critic 都只使用本体感觉，观察内容相同。
- 两者都不读取地形高度。
- 因为没有网络需要高度扫描，所以 height scanner 也会关闭。
- 同样启用 terrain-distribution curriculum，并写入 TensorBoard。

这是完全盲走的对称设置：actor 与 critic 均没有地形高度信息。

两个 task 均提供 `-Play-v0` 推理版本；另外注册了 stairs 的 play task，便于可视化评估。

## 训练

对称盲走训练示例：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-Blind-Symmetric-v0 \
  --num_envs 4096 --max_iterations 1500 --headless
```

`scripts/reinforcement_learning/rsl_rl/train_go1_blind_symmetric.sh` 封装了相同命令，但其中 Conda 路径是本机路径，换机器时需要修改。

本分支还增加了：

- `--resume_from <checkpoint>`：从指定 checkpoint 开启一个新的训练 run；
- terrain distribution 仅写 TensorBoard，避免终端日志过长；
- `analyze_gait.py`：导出 gait metrics；
- `play.py --camera_follow`：播放时相机跟随机器人。

## 实验结果

三次从零开始的训练均使用 RTX 4090、Xeon Gold 6430 的 16 核 CPU 切片、`cuda:0`、4096 个环境和每个环境 24 个 rollout steps。下表的耗时是 TensorBoard 记录的完整训练墙钟跨度，包含采样、学习和日志开销。

| 实验 | 完整训练 | 训练时长 | 结论 |
| --- | ---: | ---: | --- |
| Go1 flat | 300 iter | 约 9 分 18 秒 | 学得平地基础步态。 |
| Rough blind（非对称） | 1500 iter | 约 51 分 8 秒 | actor 仅本体感知，但可利用 privileged critic 的训练信号。 |
| Rough blind symmetric（纯盲对称） | 1500 iter | 约 53 分 39 秒 | 坡和随机 rough 能学会，但连续向上台阶是显著短板。 |

### Rough curriculum 的最终能力

为比较 terrain curriculum，将两个 1500-iter rough checkpoint 分别继续训练并记录 terrain-distribution 到 TensorBoard。以下结论来自各续训 run 的最后可用快照：非对称 blind 为 iter 1806，纯盲对称为 iter 1903。该指标是各 level 的环境占用分布，反映持续推进 curriculum 的能力；它不是固定出生点、固定指令测试下的严格成功率。

默认台阶的单级高度范围为：L0--L2 为 5.0--10.4 cm，L3--L5 为 10.4--15.8 cm，L6--L9 为 15.8--23.0 cm。

| 地形 | 非对称 blind（actor 48 / critic 235） | 纯盲对称（actor 48 / critic 48） |
| --- | --- | --- |
| 正金字塔 `pyramid_stairs` | 可稳定进入高难度 L6+。默认出生在中央高台，主要经历向外下台阶。 | 可到中高难度 L5--L7，但高难度占比更低。 |
| 倒金字塔 `pyramid_stairs_inv` | 可在坑底向外爬到中高难度，主力约 L4--L7，部分更高。 | 最显著失败项：主要停在 L0--L2，仅偶尔到 L3--L5，无法稳定进入 L6+。 |
| 方块、随机 rough、上/下坡 | 均可稳定进入高难度 L6+。 | 均可达到中高难度，主力约 L5--L7。 |

这个对照的关键点是：两组训练保存的环境、奖励、PPO 超参数与 actor 架构一致，实质差异是非对称 critic 额外接收 187 维高度扫描。高度图不会在部署时提供给 actor；它通过更准确的 value/advantage 估计降低训练信号方差，因此对离散竖直障碍的学习影响很大。

## 已上传的最终 checkpoint

模型保持原始训练日志路径：

| 实验 | Checkpoint |
| --- | --- |
| Flat Go1 | `logs/rsl_rl/unitree_go1_flat/2026-08-17_16-03-35/model_299.pt` |
| Rough blind | `logs/rsl_rl/unitree_go1_rough_blind/2026-08-18_11-02-31/model_1499.pt` |
| Rough blind symmetric | `logs/rsl_rl/unitree_go1_rough_blind_symmetric/2026-08-21_09-35-46/model_1499.pt` |

只提交上述三份最终模型；中间 checkpoint、视频、TensorBoard event 和其他日志均未提交。
