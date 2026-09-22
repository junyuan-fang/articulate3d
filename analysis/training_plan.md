# Articulate3D Challenge 打榜方案（2026-08-04）

范围：本文档只出方案 + 已核实的事实，不启动长训练。
所有"已测"数字都是我在本机跑出来的；"预估"会明确标注。

---

## 0. 结论先行

**最大的发现：不需要训练就能拿到 +18.6% 相对提升的 MAO_ST（榜单 "both" 列）。**

我用官方评测器 `benchmark/evaluate_semantic_instance.py` 在 val split 上跑了对照实验
（checkpoint = `mov_trainval.ckpt`，预测来自已有的 `results/submission_dev/mov_dev`）：

| 变体（只改后处理，mask 一个 bit 都没动） | M_ap50 | MA | MO | **MAO_ST** |
|---|---|---|---|---|
| 现状（照搬 transfer_preds_files.py 的输出） | 0.5711 | 0.4419 | 0.4502 | **0.3214** |
| + 把 axis 归一化成单位向量 | 0.5711 | 0.4419 | 0.4502 | **0.3391** (+1.77) |
| + axis 吸附到最近的正则方向（容差 15°） | 0.5711 | 0.4432 | 0.4788* | **0.3642** |
| + axis 吸附（容差 30°） | 0.5711 | 0.4662 | 0.4502 | **0.3548** |
| + axis 吸附 30° + origin 吸附到 mask 边缘（q=2%） | 0.5711 | 0.4662 | 0.4788 | **0.3812** (+5.98) |

\* 该行同时开了 origin 吸附。

- `M_ap50` 在所有变体中**逐位不变（0.5711）**——分割完全没有风险。
- MAO_ST：0.3214 → 0.3812，**相对 +18.6%**。
- 复现性检查：不做任何改动时我算出 M=0.5711 / MA=0.4419 / MO=0.4502 / MAO=0.0710 / MAO_ST=0.3214，
  与 `/data/challenge/data_download/challenge/eval_mov_val.log` 里记的
  0.57106 / 0.44190 / 0.45017 / 0.07096 / 0.32136 完全一致 → 我的评测驱动脚本是可信的。

若相对增益在 test 上同比例保持（**预估，需要 1 次 test 提交验证**）：
`0.2640 × 1.186 ≈ 0.313`，一步从 0.2640 走到 ~0.31，逼近 linxii 的 0.3360。

> 注意：val 用的 `mov_trainval.ckpt` 见过 val 场景，绝对值是乐观的。
> 但这些后处理是**几何先验**，不是拟合 val 的超参，相对增益可迁移性较高。

脚本都在 `/tmp/claude-1001/-data-challenge/c7966c1e-6406-4bad-9e6c-75683e9c5dcb/scratchpad/`：
`run_official_eval.py`（官方评测驱动）、`postproc_cache.py` + `sweep.py`（后处理扫参）、
`diag.py` + `report.py`（误差分解）、`axis_stats.py`、`origin_prior.py`、`nr_check.py`、`rescore.py`。
建议把 `run_official_eval.py` 挪进 repo 长期用。

---

## 1. 逐条核实你列的 4 个可疑点

### 1.1 origin loss 与评测判据不对齐 —— **部分误读，实际是对齐的**

事实链：

1. `scripts/train_mov.sh` 与 `scripts/infer_mov_val.sh` 都传了 `loss.regular_arti_loss=false`。
   分支选择器是 **`regular_arti_loss`，不是 `predict_articulation_mode`**（后者只选模型的 arti head 形态，
   `models/mask3d.py:546` 起是 mode 2 = pointwise origin/axis + 局部坐标）。
   所以走的是 `models/criterion.py` 的 **else 分支（几何 loss）**，
   `F.l1_loss(map_origins, target_origins)` 那条 **从来没被执行过**。你担心的逐点 L1 惩罚沿轴滑动 —— 不存在。

2. `dist_pred_axis`（GT origin 到**预测**轴线的垂距）**评测里有**。
   `benchmark/evaluate_semantic_instance.py:160-186` 的 `match_criteria_MAO_pred_standard`：

   ```python
   diff_proj_gt_axis   = diff_pred_gt - np.dot(diff_pred_gt, gt_axis)   * gt_axis   / np.linalg.norm(gt_axis)
   diff_proj_pred_axis = diff_pred_gt - np.dot(diff_pred_gt, pred_axis) * pred_axis / np.linalg.norm(pred_axis)
   is_origin = (norm(diff_proj_gt_axis) < 0.25) and (norm(diff_proj_pred_axis) < 0.25)
   ```

   而榜单 "both" 列 = **MAO_ST**（不是 MAO）。证据：我们 test both=0.2640，
   val 上 MAO_ST=0.3214 而 MAO=0.0710，比例只有 MAO_ST 对得上。
   所以 loss 里的 `dist_gt_axis + dist_pred_axis` 与 headline 指标是**一致**的。
   只有 "origin" 那一列（MO，`match_criteria_MO_pred`）单用 `dist_gt_axis`。

3. **但是**：`dist_pred_axis` 这一项在实践中几乎从不是瓶颈。val 上 180 个匹配到的 rotation 实例中，
   `d_gt<0.25` 通过率 0.694，`d_pred<0.25` 通过率 0.728，**两者同时通过 0.694**——
   即 `d_gt` 一旦通过，`d_pred` 必然通过。
   所以现在正在跑的 `train_arti_run1` 设的 `origin_loss_sym=false`（去掉该项）
   **不是修 bug，是一次温和的 loss 重加权**，风险低但收益也小，别指望它单独涨分。

**判断**：可疑点 1 里 "逐点 L1" 是误读（未启用）；"dist_pred_axis 评测没有" 也是误读（MAO_ST 有）。
真正对齐性问题在别处 —— 见 1.5。

### 1.2 26% 标注带 needsReview=true —— **误读，实际是 0 条**

我在 train+val 的 237 个场景的 `_artic.json` 里统计了全部 6349 条 articulation：

```
needsReview == true   :    0
needsReview == false  : 1667
字段缺失（key 不存在）: 4682
```

你数到的 1667 是 `needsReview: **false**`（标注员显式确认过没问题），不是 true。
再把 2316 条已进入训练/验证的 GT 实例（按 origin 精确匹配，2316/2316 = 100% join 成功）反查回 JSON：
708 条来自 `false`，1608 条来自字段缺失，**0 条来自 true**。

**判断**：过滤/降权 needsReview 这条路直接关掉，没有可过滤的数据。
（`needsReview: false` 那 708 条要不要加权？没有证据说明剩下 1608 条更差，不建议动。）

### 1.3 未用的监督信号 base / rangeMin / rangeMax / motionStates —— **真的没用，但优先级低**

`base`（父部件 pid 列表）、`rangeMin/rangeMax`、`motionStates` 确实完全没进预处理产物
（`*_articulation.h5` 只有 `axis / origin / sem_id / inter_mask`）。

- `rangeMin/rangeMax`：跨度中位数 1.571 rad（= π/2，典型 90° 开门），零跨度比例 0.000。
  它跟轴/原点是**弱**几何相关（知道转 90° 并不能定位铰链在哪条边）。做辅助损失属于正则化，不是直接增益。
- `base`：父部件层级 —— 对"哪条边是铰链"其实**强相关**（铰链在与父部件接触的那条边）。
  但要用它必须能把 pid 映射到点云 mask，这需要你还没写的 JSON→层级树解析器。
- 关键是：1.6 会说明，origin 的信息瓶颈已经被我定位成**一个 1-bit 的二选一**，
  用几何直接解就够了，不需要绕道 rangeMin/rangeMax。

**判断**：`rangeMin/rangeMax/motionStates` 列为 P3（有余力再做）；`base` 列为 P2 但要先付解析器成本。

### 1.4 train_mode / cropping —— **确认无误**

`data.train_mode="train_validation"` 才是提交 test 的配置（195+42=237 场景）；
`"train"` 只有 195 场景，但可以用干净的 val 做模型选择。
`data.cropping=true, crop_length=5.5, crop_min_size=75000` 是显存妥协项。
`conf/callbacks/callbacks_instance_segmentation.yaml` 的 ModelCheckpoint monitor 的是
`val_mean_ap_50`（分割 AP），**不是 MAO_ST** —— 如果我们的目标是关节参数，
选 checkpoint 的判据就选错了。建议改 monitor 为 `MAO_ST_ap50`。这是一个被忽略的真问题。

### 1.5 我额外找到的、你没列的对齐性问题

**(a) axis loss 用 `1 - cos`，评测用 `|cos|`。**
`criterion.py` 的几何分支：`loss = 1 - sum(normalize(pred) * normalize(target))`。
GT 轴向的符号分布（2316 条）：

| | 竖直（|z|>cos5°） | 水平（|z|<sin5°） | 其他 |
|---|---|---|---|
| rotation (1521) | 91.7% | 5.3% | 3.0% |
| translation (795) | 0.1% | 95.0% | 4.9% |

竖直轴符号高度一致（+z 1333 / −z 62）；**水平轴的面内符号基本是 50/50**
（x>0 占 0.42-0.44，y>0 占 0.42-0.45）→ 对约 36% 的实例（836/2316），
`1-cos` 在逼迫网络学一个从点云不可判、且评测根本不看的符号。
改成 `1-|cos|` 与指标严格对齐。`train_arti_run1` 已经加了 `loss.axis_loss_abs_cos` 开关，方向正确。

**(b) 训练时几何增广被完全关掉了。**
`conf/data/datasets/articulate3d_challenge_mov.yaml` 里 train/val/test 三份都是
`volume_augmentations_path: null`、`image_augmentations_path: null`，
且 `is_elastic_distortion: false`、`flip_in_center: false`、`instance_oversampling: 0`。
`datasets/semseg.py:663-668` 的翻转增广整段被注释掉了 → `flip_axis` 恒为 `{}`。
**195 个场景、零几何增广**。这是泛化的大漏洞（尤其对 axis 头，它现在只能记住场景朝向）。

⚠️ 但直接打开 `volumentations_aug.yaml` 会**灾难性地毁掉标签**：
`semseg.py:1358-1365` 只把 `shift` 施加到 origin，完全没有对 axis/origin 施加旋转矩阵。
一旦启用绕 Z 随机旋转，点云转了而 GT 轴没转 → 训练直接崩。
必须自己写显式旋转（见 P2-A）。

**(c) `semseg.py:1365` 的翻转逻辑本身是错的**（虽然当前代码路径没走到）：
方向向量在镜像下应该只翻转法向分量 `(-dx, dy, dz)`，代码写的是整体取负 `-d`。
如果将来打开翻转增广，这里必须一起修。

**(d) 评测器对预测 axis 的模长有 bug，可以合法利用。**
`match_criteria_MAO_pred_standard` 里投影写的是
`np.dot(d, pred_axis) * pred_axis / np.linalg.norm(pred_axis)`——
正确应除以 `norm**2`。当 `|pred_axis| != 1` 时垂距被算错。
模型输出的 axis 不是单位向量，`transfer_preds_files.py` 原样写盘。
**只要提交前归一化，MAO_ST 从 0.3214 → 0.3391（+1.77 AP），已实测。**
（`match_criteria_MO_pred` 用的是 GT 轴，GT 轴我验证过模长恒为 1.0，所以 MO 不受影响，
实测 MO 在两种情况下都是 0.4502，与推理一致。）

### 1.6 最重要的新发现：rotation 的 origin 是一个 1-bit 问题

用 **GT mask + GT 轴** 在 val 的 235 个 rotation 实例上测：
把 mask 点投影到垂直于轴的平面，取主方向上的两个端点作为候选原点，

| 候选 | 中位垂距 | <0.25 m | <0.1 m |
|---|---|---|---|
| mask 质心 | 0.308 | 28.1% | 1.7% |
| **两端点中较优者** | **0.008** | **100.0%** | **99.1%** |
| 两端点中较差者 | 0.612 | 2.1% | 0.9% |
| 轴对齐 bbox 四角最优者 | 0.018 | 96.6% | 93.6% |

**GT 铰链 100% 落在部件沿主方向的两个端点之一，中位误差 8 mm。**
也就是说 origin 根本不需要 3 自由度回归，它是"选左边还是选右边"的**二分类**。

在真实预测（预测 mask + 预测轴）上验证（n=180 rotation，IoU>0.5）：

| 策略 | rot origin 通过率 | 中位 d_gt |
|---|---|---|
| 现状（纯回归） | 0.694 | 0.140 |
| 吸附到最近端点（原始 min/max） | 0.656 | 0.053 |
| **吸附到最近端点（2%/98% 稳健分位）** | **0.767** | **0.034** |
| oracle（总选对边） | 0.894 | — |

稳健分位（q=2%）很关键：原始 min/max 会被 mask 漏点带偏。
oracle 还有 +12.7 个点的空间 → 这就是 P1-C（学一个"选哪边"的头）的目标。

### 1.7 val 误差分解（n=275，IoU>0.5 匹配到的 GT）

- recall@IoU0.5 = 70.5%（275/390）—— 分割上限
- axis 通过 84.4% / origin 通过 80.0%（rotation 只有 69.4%）/ MAO_ST 通过 67.6%
- 角度误差桶（0-5 / 5-10 / 10-15 / 15-30 / 30-60 / 60-90 度）= 135 / 63 / 34 / 23 / 10 / 10
- rotation 的 d_gt 桶（<.1 / .1-.25 / **.25-.5** / .5-1 / 1-2 / >2）= 72 / 53 / **31** / 17 / 3 / 4
  → **31 个是刚过阈值的"擦边失败"**，精度翻倍就能全捡回来
- 竖直 GT 轴 axis 通过 91.1%，水平 GT 轴只有 76.7% → 误差集中在水平轴的面内角
- rotation/translation 分类准确率 98.2% —— **不是瓶颈，别在这上面花时间**

### 1.8 置信度与关节正确性几乎不相关（重排序的空间）

后处理之后，n=275：

| 测试时可算的信号 | 与 MAO_ST 正确的相关系数 |
|---|---|
| 现有 score（cls_prob × mask_score） | **+0.036** |
| 回归 origin 到吸附端点的距离 | −0.360 |
| 部件在垂直于轴平面上的跨度 | −0.324 |
| 原始预测轴偏离正则方向的角度 | −0.221 |
| `score × exp(−d_edge/0.2)` | **+0.238** |

`trainer/trainer.py:763` 的 `score = scores_per_query * mask_scores_per_image` 里
完全没有关节质量项。而 AP 是按 score 排序算的：一个 IoU 高但轴错的预测会以**高分**变成 FP，
把 PR 曲线前段打烂（`evaluate_matches` 中 `found_gt` 用的就是含轴/原点的 match_criteria）。
所以重排序是真实杠杆，而且上面三个信号**测试时全都能算，不需要训练**。

---

## 2. 实验清单（按 期望收益 / 成本 排序）

成本参考：`train_arti_run1`（冻结骨干、crop 4.0 m）实测 **约 1 分钟/epoch**；
全量微调预计 1.3–1.8 分钟/epoch；val 全量评测约 4–6 分钟。

### P0 —— 零训练、零风险，先做（合计预期 test both 0.264 → ~0.31）

| # | 改什么 | 依据 | 成本 | 本地验证 |
|---|---|---|---|---|
| **P0-A** | `transfer_preds_files.py` 写盘前把 axis 归一化为单位向量 | §1.5(d)，实测 MAO_ST +1.77 | 5 分钟改代码 | `run_official_eval.py --tag x` |
| **P0-B** | 预测轴吸附到最近正则方向（\|z\|>cos τ → ±z；\|z\|<sin τ → 投到 xy 平面） | §1.5(a) GT 96.4% 在 5° 内正则；实测 τ=30° 时 MAO_ST +1.57，且 16 个变好 0 个变坏 | 30 分钟 | 同上，τ ∈ {20,30,45} 扫一遍 |
| **P0-C** | rotation 的 origin 吸附到 mask 沿主方向的 2%/98% 分位端点中较近者 | §1.6，实测 MAO_ST +2.64、中位误差 0.140→0.034 | 1 小时 | 同上，扫 q ∈ {0,1,2,5} |
| **P0-D** | 把 `run_official_eval.py` 收进 repo，作为标准离线评测入口 | 复现性 | 30 分钟 | — |

P0-A+B+C 合起来实测：**MAO_ST 0.3214 → 0.3812，M_ap50 逐位不变。**
τ=30 优于 τ=15（0.3812 vs 0.3642），45° 还没测，第一件事就是补上。

**这三条应该合成第 1 次 test 提交**（用现成的 `mov_trainval.ckpt`，不用训练）。
既锁定一个安全的分数提升，也校准"val 相对增益 → test 相对增益"的转换比例，
后面所有决策都靠这个比例外推。

### P1 —— 训练相关，高收益，中等风险

| # | 改什么 | 依据 | 成本 | 本地验证 |
|---|---|---|---|---|
| **P1-A** | axis loss 改 `1-\|cos\|`（`axis_loss_abs_cos=true`） | §1.5(a)，36% 实例符号任意 | 已实现；120 ep ≈ 2 h（冻结） | val MA_ap50 / MAO_ST |
| **P1-B** | ModelCheckpoint 的 monitor 从 `val_mean_ap_50` 改成 `MAO_ST_ap50` | §1.4，选模型判据跟目标指标不一致 | 10 分钟 | — |
| **P1-C** | **加一个"铰链在哪一端"二分类头**：候选由 mask+轴几何生成，网络只出 1 个 logit；推理时按 logit 选端点 | §1.6：oracle rot-origin 0.894 vs 启发式 0.767，纯 origin 还有 +12.7 点 | 实现 0.5–1 天 + 训练 3–4 h | val MO_ap50 / MAO_ST |
| **P1-D** | **关节质量头**：每个 query 预测 `P(axis<15° ∧ origin<0.25)`，推理时 `score × q^α`，α 在 val 上扫 | §1.8：现 score 相关性只有 0.036；先用免训练版 `score × exp(−d_edge/τ)` 试水 | 免训练版 1 h；学习版 0.5 天 + 3 h | val 同时看 M_ap50 与 MAO_ST 的取舍曲线 |
| **P1-E** | axis 头改参数化：先分类 {竖直, 水平, 其他}，水平时只回归 1 个面内角 | §1.5(a)：97.0% rotation 是竖直或水平；把 3-DoF 降成 1 分类 + 1-DoF | 实现 0.5 天 + 3–4 h | val MA_ap50 |

P1-C 是本轮**最大的单点训练收益**，因为它把 origin 从 3-DoF 回归降成 1 bit，
且几何候选的天花板是 100%/8 mm。

### P2 —— 中等收益，需要更多工程

| # | 改什么 | 依据 | 成本 | 风险 |
|---|---|---|---|---|
| **P2-A** | **加回几何增广（绕 Z 随机旋转 + 各向同性缩放），并同步旋转 origin/axis** | §1.5(b)：现在零几何增广，195 场景 | 实现 0.5 天（必须自己写显式 R，不能用 volumentations）+ 全量训练 6–10 h | 中：顺序必须是 `origin' = R(origin+shift)`、`axis' = R·axis`；写错就是标签污染。**上线前必须用一个 5 场景的 overfit sanity check 验证** |
| **P2-B** | 修 `semseg.py:1365` 的翻转轴变换（镜像只翻法向分量），然后打开 XY 翻转 | §1.5(c) | 2 h + 训练 | 中 |
| **P2-D** | 用 `base` 字段（父部件）做"哪一端是铰链"的显式监督 | §1.3 + §1.6 | 需先写 JSON→层级树解析器（1–2 天） | 中高，工期风险大 |

### P3 —— 低优先级 / 不建议

| # | 内容 | 判断 |
|---|---|---|
| P3-A | `rangeMin/rangeMax` 辅助损失 | 与轴/原点仅弱相关，做完 P1-C 后收益基本被吃掉 |
| P3-B | 过滤 needsReview | **直接取消**，§1.2 证明没有可过滤的数据 |
| P3-C | 换分割骨干 / 换 Mask3D 预训练权重 | 我们 AP50 已是榜首，纯浪费 3 周里的算力 |
| P3-D | 优化 rotation/translation 分类 | 已 98.2%，无空间 |
| P3-E | 用 `scannet200_benchmark.ckpt` 从头重训 | 2000 epoch × ~1.5 min ≈ 50 小时/次，且已有更好的起点 `mov_trainval.ckpt` |

---

## 3. 三周排期（截止 2026-08-26，test 配额剩 9 次 / 1 次每天）

**总原则：test 提交只用来（a）锁定确定收益，（b）校准 val→test 的换算比例。
所有比较、扫参、选模型全部在本地 val 上做（dev 配额 5/天、49 次，足够）。**

### 第 1 周（8/04 – 8/10）：把免费的分拿到手 + 打好训练地基

- D1（8/04–8/05）：实现 P0-A/B/C/D，在 val 上扫 τ、q。当前 `train_arti_run1` 正在占卡，
  P0 是纯 CPU 后处理，可并行。
- D2：**test 提交 #1** = `mov_trainval.ckpt` + P0 全套。
  → 得到 val 相对增益 → test 相对增益的换算系数 k。这是后面所有外推的基准。
- D2–D3：P1-B（改 monitor）；跑一个 `train_mode=train`（195 场景）的干净 baseline，
  拿到一个**没见过 val** 的模型，之后所有 val 数字才有意义。
  预算：冻结骨干 120 ep ≈ 2 h；全量微调 200 ep ≈ 5 h。
- D3–D5：实现 P1-C（端点二分类头）与 P1-D 免训练版。
- D5–D7：P1-C 训练 + val 评测；P1-A（abs-cos）单变量消融。
  **一次只动一个变量**，结果记进 §4 的表。
- 周末：**test 提交 #2** = 当周最好的组合（只在 val 上明确胜出才提）。

### 第 2 周（8/11 – 8/17）：训练侧收益

- P1-E（axis 重参数化）实现 + 消融。
- P2-A（旋转增广 + 正确的标签变换）：先做 5 场景 overfit sanity check，再全量训练。
- P1-D 学习版（关节质量头）+ 重排序 α 扫参。
- 全部在 `train_mode=train` 上比较，只看 val 的 `MAO_ST_ap50`。
- **test 提交 #3、#4**：每次只提一个已在 val 上验证过的增量组合。

### 第 3 周（8/18 – 8/26）：定版

- 8/18–8/21：把 val 上胜出的组合用 `train_mode=train_validation`（237 场景）重训 —— 提交版必须吃满数据。
  预算：全量微调 250 ep ≈ 6–7 h/次，最多排 3 次。
- 8/22–8/24：**test 提交 #5、#6、#7**，做最终的后处理超参微调（τ、q、α）。
- 8/25：**#8** = 最佳配置的最终确认。
- 8/26：**#9** 留作应急（发现格式/回归问题时的回滚位）。

**配额分配：1 次基线校准 + 5 次增量验证 + 2 次定版 + 1 次应急 = 9 次。**

---

## 4. 结果记录表（每跑完一条就填，禁止填没实际跑出来的数）

| 实验名 | 配置路径 | split | seed | git state | M_ap50 | MA | MO | MAO_ST | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| baseline-asis | mov_trainval.ckpt | val | – | – | 0.5711 | 0.4419 | 0.4502 | 0.3214 | 与官方 log 逐位一致 |
| pp-norm | 同上 + P0-A | val | – | – | 0.5711 | 0.4419 | 0.4502 | 0.3391 | 已测 |
| pp-ax30 | 同上 + P0-B(τ=30) | val | – | – | 0.5711 | 0.4662 | 0.4502 | 0.3548 | 已测 |
| pp-ax15-or | + P0-B(15)+P0-C(q=2) | val | – | – | 0.5711 | 0.4432 | 0.4788 | 0.3642 | 已测 |
| **pp-full** | **+ P0-B(30)+P0-C(q=2)** | val | – | – | **0.5711** | **0.4662** | **0.4788** | **0.3812** | **已测，当前最佳** |
| test-baseline | mov_trainval.ckpt | test | – | – | 0.4370 | 0.3801 | 0.3160 | 0.2640 | 榜单已有 |

---

## 5. 风险分级

### 安全（M_ap50 不可能变，已实测逐位不变）
- P0-A 归一化 axis
- P0-B axis 吸附（16 个变好、0 个变坏；理论上最坏情况是 GT 轴离正则方向 >15° 的那 3.6% 实例，
  把 τ 卡在 30° 以内可把最大诱导误差限制在 30°，实测没有一例回退）
- P0-C origin 吸附（q=2 稳健分位版；q=0 原始 min/max 版实测会掉 rot-origin 0.694→0.656，**别用 q=0**）
- P1-B 换 checkpoint monitor

### 低风险
- P1-A `1-|cos|`（与指标严格对齐；唯一副作用是 cos=0 处不可导，实践中无碍）
- P1-D 免训练重排序（会牺牲一点 M_ap50 换 MAO_ST，取舍曲线在 val 上可见，α 可调回 0）
- 冻结骨干微调（`freeze_except_articulation=true`，分割输出与源 ckpt 完全一致）

### 中风险（必须先做 sanity check）
- **P2-A 旋转增广**：写错变换顺序 = 标签污染 = 全盘崩。上线前必须：
  (1) 5 场景 overfit 到 loss≈0；(2) 直接可视化增广后的 axis/origin 与点云是否一致。
- **P2-B 翻转增广**：`semseg.py:1365` 的 bug 必须一起修，否则等于给 axis 喂随机标签。
- P1-E axis 重参数化：分类错误会造成 90° 的灾难性误差，需要 fallback 到回归值。

### 高风险（不建议在 3 周内做）
- 换骨干 / 从 scannet200 从头训练（P3-C/E）：50+ 小时/次，且我们分割已经是榜首。
- P2-D 层级树解析器：工期不可控。
- 任何"多假设提交"（同一个 mask 提交多个轴假设）：多余的假设一律计为 FP，
  `evaluate_matches` 里没有"重叠已匹配 GT 就免罚"的豁免，**会同时打掉 M_ap50 和 MAO_ST**。

---

## 6. 立即可执行的下一步

1. 等 `train_arti_run1`（当前占卡，PID 1752102，`freeze_except_articulation=true`,
   `axis_loss_abs_cos=true`, `origin_loss_sym=false`, 120 ep）跑完，用它的 ckpt 跑一次干净 val。
   注意它是从 `mov_trainval.ckpt` 起步、`data.train_mode=train`，
   所以它的 val 数字**仍然被污染**（初始权重见过 val），只能看训练是否稳定，不能当作选型依据。
2. 并行实现 P0-A/B/C（纯 CPU），扫 τ ∈ {20,30,45}、q ∈ {0,1,2,5}。
3. 出 test 提交 #1，拿到 val→test 的换算系数。
4. 起一个真正干净的 baseline：`data.train_mode=train` + 从 `scannet200_benchmark.ckpt`
   或 `mov_trainval.ckpt` 起步但**不看 val 选模型**，作为第 2、3 周所有消融的对照。
