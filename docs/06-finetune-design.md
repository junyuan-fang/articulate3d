# 关节参数头 v2：设计、消融与结论

> 目标：在**不动分割**的前提下，把关节参数（axis / origin）的保留率拉上去。
> 判据只有一个：`AP50_axis_origin`（榜单的 `MAO_ST_ap50`）。
> 所有 val 分数由 `analysis/score_pkl.py` 用官方判据算出，与服务器逐位一致。

## 0. 一句话结论

| | val AP50 | axis | origin | **both** |
|---|---|---|---|---|
| baseline（`mov_trainval.ckpt`） | 0.5716 | 0.4432 | 0.4495 | **0.3215** |
| 免训练后处理（修正版） | 0.5716 | 0.4677 | 0.4631 | **0.3828** |
| *（占位：v2 训练结果见第 5 节）* | | | | |

**这一轮最重要的单条发现不是模型，而是一个被写进 `05-improvement.md` 却从未端到端验证的后处理规则是错的**，
修正后免训练就有 `0.3215 → 0.3828`（+0.061），详见第 1 节。

---

## 1. 先修 bug：origin 端点吸附的方向选错了

`05-improvement.md` 的第 0 层 ③ 写着「origin 吸附到 mask 端点（2%/98% 分位）」，
声称 val 能到 0.3812。`analysis/postprocess2.py` 按「部件**第一主方向**的两端」实现了它。
端到端跑一遍（`analysis/pp_sweep.py`）：

```
norm+canon                        AP50 0.5716  axis 0.4677  origin 0.4495  BOTH 0.3550
norm+canon+endpt(1st PCA)         AP50 0.5716  axis 0.4677  origin 0.2619  BOTH 0.1898   ← 掉 0.19
norm+canon+endpt(perp, 5/95)      AP50 0.5716  axis 0.4677  origin 0.4631  BOTH 0.3828   ← +0.028
norm+canon+endpt(perp, 0/100)     AP50 0.5716  axis 0.4677  origin 0.4209  BOTH 0.3435
norm+canon+nearest point          AP50 0.5716  axis 0.4677  origin 0.4732  BOTH 0.3744
```

**原因是几何的**：一扇绕竖直轴转的门，最长方向是竖直，第一主方向的两端是门的**上沿和下沿**；
而铰链在门的**侧边**。正确的方向是「**垂直于轴**的最大延展方向」。
原始 oracle 结论（「铰链 100% 落在端点」）是**在给定 GT 轴的条件下**成立的，
这个条件在复现实现里被丢掉了。

修正已写回 `analysis/postprocess2.py`（含 5%/95% 分位，比 2%/98% 略好）。
**端到端跑一遍这个脚本本身**（不是 sweep 里的等价实现），确认提交路径可信：

```
python postprocess2.py <val preds.pkl> out.pkl <points_dir>   # steps=normalize,canonical,endpoint
python score_pkl.py out.pkl
  AP50 0.5716   AP50_axis 0.4677   AP50_origin 0.4631   AP50_axis_origin 0.3828
```

`AP50` 与 baseline 逐位相同（后处理不碰 `pred_masks`），分割零风险。

> 教训：`05-improvement.md` 里 ①②③ 三条只有 ① 和 ② 被端到端验证过，③ 是从 oracle 分析外推的。
> 外推的那条恰好是三条里唯一错的。

## 2. origin：问题的全部是「在自己的 mask 里选点」

`analysis/pp_sweep2.py` 用 GT 做了两条上界（**仅用于估算收益，绝不提交**）：

| 干预 | AP50_origin | **both** | 说明 |
|---|---|---|---|
| norm+canon（不动 origin） | 0.4495 | 0.3550 | |
| + 端点(perp)，**用预测 origin 选边** | 0.4631 | 0.3828 | 现在能做到的 |
| + 端点(perp)，**ORACLE 选边** | 0.4963 | 0.4147 | 二选一选对 = +0.032 |
| + **mask 内最优点**（ORACLE） | 0.5473 | **0.4439** | 中位误差 **6 mm**，175/175 全部 < 0.25 m |

**最后一行是这次设计的全部依据**：
只要允许在**模型自己预测的 mask** 内自由挑一个点，origin 判据就 100% 通过。
也就是说 origin 头不缺表达能力、不缺特征、不缺分辨率 —— 它缺的是**选择机制**。

而现在的 mode 2 头（`models/mask3d.py`）做的恰恰是「不选择」：

```python
origin_per_point = origin_embed_head(feat) + coordinates     # 逐点，与 query 无关
outputs_origins  = einsum("bnq,bnd->bqd", mask, origin_per_point) / num_masked
```

**对整个 mask 取均匀平均 = 取质心**。诊断里的两条数字正是这个公式的直接后果：
77.2% 的 origin 被拉向质心、只回收了质心→铰链距离的 29%；
误差与部件尺寸无关（Spearman 0.006），因为它是「平均」这个算子的偏差，不是尺度问题。

### 改动：query 条件化的 masked softmax 聚合

```python
logits = einsum("bnd,bqd->bnq", W_out·key(feat_n, coord_n), query(q)) / sqrt(d)
       + bias(feat_n, coord_n)
logits = logits.masked_fill(~mask, -1e4)
attn   = softmax(logits, dim=points)          # 每个 query 在自己 mask 内归一
origin = einsum("bnq,bnd->bqd", attn, origin_per_point)
```

设计要点，每条都对着一条证据：

| 设计 | 依据 |
|---|---|
| **softmax 而不是均值** | 上表最后一行：选点就够了 |
| **query 条件化**（q·k） | 原头的逐点偏移与 query 无关，同一个点在两个 mask 里只能给同一个答案；「我是不是这个部件的边缘」必须知道是哪个部件 |
| **key 里拼进坐标** | 「在不在边缘」是位置问题 |
| **`W_out` 与 `bias` 零初始化** | logits ≡ 0 ⟹ softmax = 1/N ⟹ **精确等于原来的均值**。加载 checkpoint 后输出逐位不变，微调只能是主动离开 baseline，不会静默掉分 |
| **只物化 (b, n, q) 张量** | 原实现物化 (b, n, q, 3)，100k 点 × 100 query 时 24 GB 卡 OOM；这条沿用已验证的 einsum 改写 |
| 输出是 mask 内点的**凸组合**（+ 逐点偏移） | 天然落在部件上，正好对上「吸到最近点就白赚 +0.048」那条 |

**零初始化等价性有两处验证**：
- 单元测试 `analysis/test_head_v2_identity.py`：`max |attn − mean| = 3.1e-7`，空 mask 不产生 NaN，且梯度确实能到达零初始化的参数（否则头永远训不动）；
- **真实数据端到端**：把 `origin_agg=attn` 的模型加载 `mov_trainval.ckpt` 直接推 val，
  得 `0.5714 / 0.4423 / 0.4505 / 0.3217`，对 baseline `0.5716 / 0.4432 / 0.4495 / 0.3215`
  ——全部落在文档记录的推理噪声（AP50 ±0.0005，axis ±0.0013）之内。

### 为什么梯度方向是对的（不是玄学）

损失 `L = ‖(o − g) × a_gt‖`，`o = Σ_n w_n p_n`。记 `u = ∂L/∂o`（垂直于轴、指离 GT 直线的单位向量），则

```
∂L/∂logit_m = w_m · ((p_m − o) · u)
```

即：**位于「o 朝 GT 直线那一侧」的点，权重被按比例调高**。
这正好是一步 mean-shift。而且损失只惩罚垂距，
所以它不会强行去选端点，只会把加权均值推到铰链**直线**上 —— 与评测判据完全同构。

## 3. axis：先证明「分类」比「回归」对，再选 anchor

诊断给的是一个非常干净的失败模式：93.3% 的 rotation GT 轴竖直，
自由回归 3-vector 在 `1−cos` 损失下塌陷到重力，8 个横向铰链 100% 全错（85–90°），
且「永远输出 z」的平凡基线在 rotation 上 **9 胜 0 负**。
自由回归的病根是它可以输出「折中方向」；离散选择不能。

### anchor 集合是量出来的，不是拍的

bin+residual 头在初始化时（两个输出层零初始化 + logits 被 `|cos(anchor, free_axis)|` 偏置）
的行为，等价于对 baseline 的 `pred_axises` 做一次纯后处理。
所以可以在 CPU 上把每种 anchor 方案的「初始化行为」直接算出分数（`analysis/pp_sweep3.py`）：

| anchor 方案 | AP50_axis | both |
|---|---|---|
| 不吸附（仅归一化） | 0.4432 | 0.3393 |
| z + 4 个方位角（45° 间隔），硬吸 | 0.4485 | 0.3367 |
| z + 4 个方位角，22.5° 阈值 | 0.4442 | 0.3323 |
| z + 8 个方位角，硬吸 | 0.4655 | 0.3537 |
| **{重力, 自身的水平投影}，硬吸** | **0.4677** | **0.3560** |
| 参考：分类别 canonical 吸附(30°) | 0.4677 | 0.3550 |

我最初写的是「z + 4 方位角」，**它比不吸附还差**。
原因很清楚：translation 轴是水平的但**方位角是任意的**，
固定方位角网格把 translation 分支唯一擅长的那个自由度量化掉了。

所以最终 anchor 只有两个，其中一个是**数据相关**的：

```
anchor_0 = (0,0,1)                                   固定，重力
anchor_1 = normalize(free_axis − (free_axis·z) z)    自身水平投影，方位角精确保留
axis     = normalize(anchor_c + residual_c),  c = argmax(logits)
logits   = T·|cos(anchor, free_axis)| + cls_head(query ⊕ mask-pooled feat)
```

### 为什么不需要改 `criterion.py`

前向用 `argmax` 硬选，反向用 **straight-through**（`w = hard + p − p.detach()`）。
于是既有的、已经对齐评测的 `1 − |cos|` 损失可以直接训练这个分类器，
不需要新的 CE 项、不需要把 logits 透传到 loss、不需要动 Hungarian matcher。
单元测试确认梯度确实到达 `axis_cls_head`。

`axis_prior_temp=10` 让未训练的分类器**精确复现上表 0.4677 那一行**，
也就是说 axis 头的起点已经比 baseline 高，训练是在这个起点上继续。

### ⚠️ 但这个头训出来是失败的，已放弃（见第 6 节）

跑了 9 个 epoch（`results/v2_axisbin_unbounded_partial/`），趋势非常清楚：

| | rot_MA | rot_MO | rot_BOTH | tr_MA | **tr_BOTH** | **avg BOTH** |
|---|---|---|---|---|---|---|
| baseline | 0.5734 | 0.4720 | 0.3323 | 0.3104 | 0.3104 | 0.3215 |
| epoch 4 | 0.5820 | 0.4560 | 0.3690 | 0.2440 | 0.2440 | 0.3060 |
| epoch 9 | **0.6090** | 0.4620 | **0.3940** | **0.1580** | **0.1580** | 0.2760 |

- rotation 确实变好（0.332 → 0.394）
- 但 **translation 塌了**（0.310 → 0.158），而 translation 占分一半，净结果比 baseline 差

而且 rotation 那点收益**并不是赚来的**：免训练的重力吸附后处理本来就能把 rot_MA 做到 0.6165
（见 `04-diagnosis.md` 干预 A），比这里训出来的 0.609 还高一点。
**即：这个头在 rotation 上只是把免费的后处理重新学了一遍，代价是毁掉 translation。**

原因定位到 `cands = normalize(anchor + res)` 里 **`res` 无界**：
残差足够大时归一化后可以指向任意方向，anchor 结构名存实亡，
等于退回自由回归，只是起点更好 —— 而自由回归的病根正是它能输出折中方向。
translation 的水平 anchor 本来精确保留方位角，被残差推歪后就没救了。

已加 `model.axis_res_scale`（`res = scale * tanh(res)`，0.3 约等于限制在 17° 以内），
默认 0.0 = 保持这次实验的无界行为。**尚未验证**。

结论：**当前提交路径不用这个头**，axis 交给已验证的后处理。

## 4. 训练配方

| 项 | 值 | 理由 |
|---|---|---|
| 起点 | `ckpts/checkpoints/mov_trainval.ckpt` | 头从 baseline 质量起步而不是从零 |
| `general.freeze_except_articulation` | true | 分割分支一个参数都不动；实测 trainable 166,797 / 39,629,511 = 0.42%。⚠️ **单靠这个 AP50 仍会变**——BN buffer 不受 `requires_grad` 约束，必须配合 BN `eval()` 冻结，见下方 BN 一节 |
| `loss.axis_loss_abs_cos` | true | 评测取 `|cos|`，反向 180° 是满分命中；原 `1−cos` 给它最大惩罚 |
| `general.new_head_lr_mult` | 5.0 | 判别式学习率：v2 的新模块是**故意零初始化**的、要走很远，而 `origin_embed_head` / `axis_embed_head` 已收敛只需微调 |
| `callbacks=callbacks_articulation` | monitor `MAO_ST_ap50` | 原本 monitor `val_mean_ap_50`，冻结骨干后它基本是常数，等于随机选 epoch |
| scheduler | OneCycleLR，`max_lr=[2e-4, 1e-3]` | 两个参数组各一个峰值 |
| crop | 4.0 m / min 30k | 单卡 24 GB，约 1.4 s/it × 195 it ≈ 4.5 min/epoch |

**踩到的坑**（已修，记录以免重犯）：
- 把 `max_lr` 写回 OmegaConf 会变成 `ListConfig`，`OneCycleLR` 在 `max_lr / div_factor` 处炸，
  且被 Hydra 包成 `HydraException` → 再被 Hydra 1.0 的 `print_exception(etype=...)` 二次异常盖住，
  日志尾巴只剩一句 `unexpected keyword argument 'etype'`。必须传**原生 Python list**。
- `kill_runs.sh` 的进程模式匹配的是绝对路径，而脚本 `cd` 之后 argv 是相对路径，**一个都没杀到**；
  而且 PL 的 dataloader worker 会把 comm 改成 `pt_data_worker`、父进程死后被 init 收养，
  继续占着 19 GB 显存。两处都已修。
- 打开 `train_origin_only` 之后训到第 13 个 iteration 崩：
  `RuntimeError: element 0 of tensors does not require grad`。
  因为此时**唯一可导的项只剩 rotation 的 origin 损失**，
  而 `criterion.py` 在「没有匹配上」和「translation」两个分支里 append 的是
  `torch.tensor(0.0)` —— 一个没有 grad_fn 的叶子。
  只要某个 crop 匹配到的实例全是 translation，整个 batch 的 loss 就成了常数。
  改成 `pred.sum() * 0.0` 保住计算图，数值完全不变。
  （这个 bug 在原配方下永远不会触发，因为 mask/ce 分支一直可导。）

## 5. 结果

所有数字都是 val，判据与服务器逐位一致。
`baseline` / `+postproc` 两行由 `analysis/score_pkl.py` 算出；
训练中的行由 trainer 自己的 `what_*` 表读出（`analysis/parse_train_log.py`），
两者在 baseline 上对得上（trainer 0.322 vs harness 0.3215）。

### 5.1 训练轮次一览（全部 `freeze_except_articulation=true`，AP50 恒为 0.5716）

| 运行 | 训练了什么 | axis | origin | **BOTH** | rot_MO | tr_MA |
|---|---|---|---|---|---|---|
| baseline `mov_trainval.ckpt` | — | 0.4432 | 0.4495 | 0.3215 | 0.4720 | 0.3104 |
| **baseline + 修正后处理** | — | 0.4677 | 0.4631 | **0.3828** | — | — |
| `v2_axisbin_unbounded_partial` @ep9 | origin attn + **axis bin** + 原 axis 头 | 0.3830 | 0.4660 | 0.2760 | 0.4620 | **0.1580** |
| `v2_origin_axisdrift_partial` @ep4 | origin attn + 原 axis 头 | 0.4350 | **0.4730** | 0.3330 | **0.5060** | 0.2930 |
| `v2_origin_only` | **只训 origin 分支** | *（进行中）* | | | | |

两条已经能下的结论：

1. **origin attention 有效**：`v2_origin_axisdrift_partial` 才 5 个 epoch，
   `rot_MO` 就从 0.4720 → **0.5060**（+0.034），origin 整体 0.4495 → 0.4730。
   这正是第 2 节 oracle 预测的方向。
2. **axis bin 头无效且有害**，见第 3 节末尾。已从提交路径移除。

顺带发现：即使只是让原来的 `axis_embed_head` 跟着一起训（`axis_loss_abs_cos=true`），
axis 也会**掉**（0.4432 → 0.4350，tr_MA 0.3104 → 0.2930）。
所以最终配方加了 `general.train_origin_only=true`，可训参数缩到 87,559 个
（`origin_embed_head` + `origin_attn_*`，`feature_arti_emb` 也冻掉，因为它同时喂 axis 分支）。
**（前提是 BN 也被冻住；否则下面这句不成立）** 这样 mask 和 axis 才与源 checkpoint 逐位相同，`AP50` 和 `AP50_axis` 才不会变，
`AP50_axis_origin` 的任何变化都只能归因于 origin 聚合方式。**

### 5.2 最终评测

*（`v2_origin_only` 跑完后填入：raw + 后处理，以及 `origin_agg=mean` 的推理期消融）*

## 6. 失败与放弃的尝试

| 尝试 | 结果 | 判断 |
|---|---|---|
| origin 端点吸附用第一主方向 | both 0.3550 → **0.1898** | 几何选错，已改为「垂直于轴的最大延展方向」 |
| 端点用 0/100 分位（真 min/max） | both 0.3435 vs 5/95 的 0.3828 | 离群点主导，必须用稳健分位 |
| axis anchor = z + 4 个固定方位角 | axis 0.4485，比不吸附的 0.4432 几乎没涨 | 量化掉了 translation 的方位角；改成数据相关的水平 anchor |
| axis anchor 加角度阈值（只在接近时才吸） | 反而更差（0.4442 / 0.4445） | 阈值把「模型本来就错的那些」保留了下来，而这些正是要被 anchor 救的 |
| 给 origin 头加密集 vote 监督（VoteNet 式） | **未做** | 需要把 attention 权重透传到 criterion；先看直接反传够不够，不够再加 |

## 7. 复现

```bash
source /data/challenge/USDNet/env.sh

# 单元测试：零初始化等价性 + 梯度可达性
PYTHONPATH=/data/challenge/USDNet python /data/challenge/analysis/test_head_v2_identity.py

# 训练
cd /data/challenge/USDNet
EXP=v2_full SAVE_DIR=./results/v2_full ORIGIN_AGG=attn AXIS_BIN=true \
  EPOCHS=45 VAL_EVERY=5 LR=0.0002 NEW_HEAD_LR_MULT=5.0 \
  bash scripts/train_arti_v2.sh

# 评测（本地 val 与服务器逐位一致，不消耗提交配额）
CKPT=./results/v2_full/<best>.ckpt SAVE_DIR=./results/eval_v2_full \
  ORIGIN_AGG=attn AXIS_BIN=true bash scripts/infer_mov_val_v2.sh
PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis \
  python /data/challenge/analysis/score_pkl.py \
  ./results/eval_v2_full/debug/val_preds/preds.pkl

# 后处理扫描（全部是纯 CPU，改 axis/origin 不动 mask，AP50 逐位不变）
cd /data/challenge/analysis
PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis python pp_sweep.py   # origin 规则
PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis python pp_sweep2.py  # oracle 上界
PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis python pp_sweep3.py  # axis anchor
```

## 8. 注意事项

- **val 分数偏乐观**：`mov_trainval.ckpt` 在 train+val 上训练过。v2 的头只在 `train` 上微调
  （`TRAIN_MODE=train`），但冻结的骨干/decoder 见过 val，所以 mask 是乐观的。
  所有对比都在同一 mask 上做，相对比较有效；绝对值不能当 test 估计。
- **推理不确定**（`general.seed: null`）：< 0.002 的差异是噪声。
- 用 `MAO_ST_ap50` 选 checkpoint 又用 val 报分，存在选择性偏差；下面同时报「选中 epoch」和「最后 epoch」。
- 设计依据是上面的诊断数据加通用做法（mask-attention pooling、bin+residual 朝向回归、
  straight-through 离散选择、判别式学习率），没有引用具体论文的结论。

---

## 9. ⚠️ BatchNorm buffer 漂移 —— 第 5 节的 v2 结论已被推翻

`requires_grad_(False)` 只冻**参数**。BN 的 `running_mean` / `running_var` 是 **buffer**，
只要模块处于 `train()` 模式，每次前向都会被刷新。
对比 `mov_trainval.ckpt` 与 `v2_origin_only@ep4`（`analysis/check_ckpt_diff.py`）：

```
BN buffer  124 个 -> 124 个全部漂移
非 BN 参数  306 个 ->   6 个变化，全部属于 origin_embed_head    <- 参数冻结本身是对的
```

### 对照实验：同一份权重，只差 BN running stats 是否恢复

| | AP50 | axis | origin | **both** |
|---|---|---|---|---|
| baseline `mov_trainval` | 0.5711 | 0.4419 | 0.4495 | **0.3215** |
| v2 ep4 **BN 漂移** | 0.5720 | 0.4558 | 0.4513 | **0.3369** |
| v2 ep4 **BN 恢复** | 0.5711 | 0.4419 | 0.4486 | **0.3215** |

**结论：v2 那 +0.0154 全部来自 BN 漂移，不是 origin attention 头。**

BN 恢复后 `both` 精确回到 baseline 的 0.3215，`AP50` 与 `AP50_axis` 逐位相同，
`AP50_origin` 甚至比 baseline **略低**（0.4486 vs 0.4495）——
训了 4 个 epoch 的 origin 头**没有带来任何 origin 改善**。

所以第 5 节「origin attention 有效，`rot_MO` 0.4720 → 0.5060」这条结论**作废**。
BN 漂移顺带把 axis 也抬了（0.4419 → 0.4558），但那是副作用：不可控、不可复现，
而且统计量漂向 4 m crop 而推理跑整场景，是**反向**的 train/test 失配。

### 修法

被冻结的 BN 全程保持 `eval()`。Lightning 每个 epoch 都会重新调 `.train()`，
所以必须覆写 `LightningModule.train()`，不能一次性 `.eval()`。已实现在 `trainer/trainer.py`。
关节头全是 Linear/ReLU 不含 BN，一刀切冻结所有 `_BatchNorm` 正是想要的行为。

### 教训

**「冻结」有歧义：冻参数 ≠ 冻模块行为。** 除 BN 外，dropout 以及任何带 running buffer
的层都有同类问题。做「保住 A 只改 B」的实验时，判据应该是**端到端比对两个 checkpoint
的 state_dict**，而不是相信 `requires_grad` 的语义。

## 10. 当前进行中：`v3_origin_bnfix`

BN 修复后的正式 scale up —— **第一轮归因干净的实验**。

```
只训 origin 分支（87,559 参数）+ BN 全程 eval()
150 epoch   crop 5.5m/75k（原始配方，OOM 已修）   lr=2e-4   new_head_lr_mult=5
每 5 epoch 验证，按 MAO_ST_ap50 保留 top-5
约 10.4 分钟/epoch -> 满程约 26 小时
```

判据：**若前 30 个 epoch 完全没有起色，说明 origin attention 这条路本身有问题**，
应换方向而不是继续等。

## 12. v4 崩溃复盘 + Stage A (v5_res) 终局

**v4**（三头联训 + 双层再平衡）：init 0.3743 → ep4 0.089，25 epoch 平台 0.15。根因分析见
[07](07-sota-options.md) 引用的算术：hbal×3 与 loss×5 叠乘把训练世界的竖直:水平比推到约 1:1.4，
而真实世界是 13:1 —— 直通 argmax 硬决策下，先验错 10 倍 = 大量竖直铰链被误翻。教训：**再平衡只用一层，一次只动一个变量**。

**Stage A / v5_res**（只训 res 头，scale=0.1，风险钳制）：
- 60 epoch 跑完，最好 ep53 raw **0.3787**（init 0.3743，+0.0044，刚过判定线）——钳制生效，没崩
- 但 **+几何栈后 0.4219 < ep19+几何栈 0.4306**：res 打磨的方位角正是 PCA 接管的地盘，
  且其微调过的轴轻微干扰 PCA 一致门与端点方向
- **终局判定：不采用**。几何规则取代了这条训练路线

### ⚠️ 评测配置陷阱（浪费了一次完整评测）

第一次评 ep53 时 axis 崩到 0.027，一度以为 checkpoint 坏了。实际是
`infer_mov_val_v2.sh` 默认 `axis_res_scale=0.0`——**0.0 走的是"残差不过 tanh 裸加"分支**，
而训练在 0.1 有界下进行。zero-init 的 ep19 评测时没暴露（0 怎么加都是 0）。

**教训：hydra 超参不存进 checkpoint。带行为开关的模块（res scale、agg 模式、bin 开关），
评测命令必须和训练命令逐项对齐；对齐的验证方法 = 离线复算 trainer 自报的分数**
（本例修正后 0.3787 逐位一致）。

### 中途结果（raw，未叠加后处理）

```
baseline              0.3215
epoch  9              0.3447
epoch 14              0.3433
epoch 19              0.3477   <- 目前最好，+0.0262
epoch 24              0.3298
epoch 29              0.3301
```

**BN 冻结后 origin attention 确实有效**：+0.0262 远超 ±0.002 噪声带。
与 v2（去掉 BN 漂移后精确回到 0.3215、零增益）形成明确对比 —— 这次的增益归因于 origin 头。

epoch 19 后回落，疑似过拟合；150 epoch 大概率跑不满，最终选的会是 20 上下的 checkpoint。

## 11. crop 尺寸与显存：为什么停在 4.0 m

5.5 m 会 OOM（炸在 `MinkowskiConvolution.forward`）。曾考虑「只对会 OOM 的场景做优化」，
实测否掉了这个前提 —— **不是少数场景，是大多数**：

```
窗口      中位点数    p90       max
4.0 m    181,518   234,128   342,403
5.5 m    278,491   359,100   547,691

点数 >30万的场景:  4.0m  2/195      5.5m  61/195
点数 >40万的场景:  4.0m  0/195      5.5m   9/195
```

5.5 m 下 31% 的场景会产出 30 万点以上的 crop，中位数就有 27.8 万。

### 梯度检查点在这里救不了

它省的是「为反传保留的激活」。但骨干和 decoder 已全部冻结（`requires_grad=False` + BN eval），
输出不带计算图、**本来就不存激活**。占显存的是另外两块：

- 头部与损失里的 `(B, N, Q)` 张量 —— N=54.8 万、Q=100 时单个 219 MB，
  而 Hungarian 匹配 + 3 个 decoder 的 aux loss 会有好几份
- MinkowskiEngine 稀疏卷积前向的临时缓冲（OOM 正是炸在这里）

对冻结骨干做检查点，省不到不存在的激活。

### 结论：4.0 m 够用

- **最大可动部件 3.50 m，4.0 m 窗口 100% 装得下**（部件最大边长中位 0.80 m / p90 2.16 m / p99 3.49 m），
  不存在「部件被切断导致端点算错」的问题
- 骨干和 decoder 全冻，上下文多少不影响特征编码；唯一在学的 87.6 K 参数只吃掩码内的逐点特征
- 5.5 m 多出来的点绝大部分是更远处的墙和地板，只经过一个不再更新的编码器，对 origin 头的梯度无贡献

真要用 5.5 m，对症的做法是**限制单 crop 点数上限**（超过就下采样），而不是梯度检查点。
但预期收益接近 0，未做。若要确证，可在 v3 收敛后用同一份权重在 4.0/5.5 上各再微调 20 epoch 做 A/B。

### 教训

**短测试不能当长跑验证。** 5.5 m 的 OOM 测试只跑了 2 个 epoch 就判定通过，
但峰值显存取决于 195 个场景里**最大的那个 crop**，短测试根本碰不到。
Epoch 0 跑到 97% 才炸。
