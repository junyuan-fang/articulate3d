# 关节参数误差诊断

**一句话结论：axis 是「长尾 + 塌陷到重力先验」，origin 是「整体精度不足」。两者病因完全不同，改法也不同，且几乎独立（φ = −0.11），收益可加。**

脚本与原始数据：`/data/challenge/analysis/`，汇总 `SUMMARY_arti_diagnosis.json`，逐实例 275 行记录在 `arti_error_diagnosis.json` 的 `rows` 字段。

## 评测判据（务必先搞清楚）

源码 `benchmark/evaluate_semantic_instance.py:101-175`。四个指标**都是 AP@IoU50，只是 TP 判据不同**：

| 指标 | TP 判据 |
|---|---|
| `AP50` | mask IoU > 0.5 |
| `AP50_axis` | IoU>0.5 **且** `arccos(\|cos(pred,gt)\|) < 15°` |
| `AP50_origin` | IoU>0.5 **且** origin 距离 < 0.25 m |
| `AP50_axis_origin` | 三者全满足 ← **榜单指标** |

**三个最容易理解错的地方：**

1. **origin 距离不是点到点距离**，是投影掉沿轴分量后的垂直距离。物理上是对的——旋转轴是**直线**，轴上任何一点都是合法 origin，沿轴滑动不该被惩罚。

   这正是 `MAO_ap50`(0.071) 与 `MAO_ST_ap50`(0.321) 差距悬殊的原因：前者用朴素的 `norm(origin_ref - origin_checked) < 0.25`，要求 origin 点本身重合。**服务器用的是后者。**

2. **⚠️ 榜单指标 `MAO_ST` 要求两个投影都通过，其中一个用 `pred_axis`：**
   ```python
   diff_proj_gt_axis   = d - dot(d, gt_axis)   * gt_axis   / norm(gt_axis)
   diff_proj_pred_axis = d - dot(d, pred_axis) * pred_axis / norm(pred_axis)   # ← 别漏了这项
   is_origin = (norm(diff_proj_gt_axis) < 0.25) and (norm(diff_proj_pred_axis) < 0.25)
   ```
   （`AP50_origin` / `match_criteria_MO_pred` 只用 gt_axis，两者不要混淆。）

   **正交投影本该除 `norm²`，这里除的是 `norm`。** GT 轴模长恒为 1 不受影响，但**模型输出的轴不是单位向量**（实测模长 0.11~1.46），所以：

   > **把 `pred_axises` 归一化成单位向量会改变判定结果，实测白拿 val +0.0177（0.3214 → 0.3391），`AP50` 一字不变。**

   推论：`criterion.py` 里的 `dist_pred_axis` 项**对应评测里的 `diff_proj_pred_axis`，不是多余项**。（不过实测它几乎从不是瓶颈——`d_gt` 通过时 `d_pred` 必通过，两者都是 0.694。）

3. **translation 类跳过 origin 判据**（`if ID_TO_LABEL[arti_label] == 'translation': return is_MA`）。合理，平移没有轴心。

轴的角度判据取 `abs(cos)` 且内部归一化，**所以未归一化不影响 `AP50_axis`** —— 但如上，**影响榜单指标 `AP50_axis_origin`**。

## ⚠️ 仓库的 `runner.test()` 永远算不出 AP

`test_epoch_end(outputs, mode='test')` 的 `mode` 默认是 `'test'`，而 `eval_instance_epoch_end` 里整个 AP 计算被 `if mode != 'test'` 挡住。**无论 `data.test_mode` 设成什么都只输出 loss。**

AP 只在 PL 的 validation 循环里算（`validation_epoch_end` → `test_epoch_end(mode="validation")`），而仓库的 main 只暴露了 `fit` 和 `test`。**所以我加了 `eval_val.py` 调 `runner.validate()`。**

## harness 可信度

`analysis/build_matches.py` 直接复用官方 `assign_instances_for_scan` / `evaluate_matches`，逐位复现 val 分数：

| | harness | 服务器 |
|---|---|---|
| AP50 | 0.5711 | 0.5711 |
| AP50_axis | 0.4419 | 0.4419 |
| AP50_origin | 0.4502 | 0.4502 |
| AP50_axis_origin | 0.3214 | 0.3214 |

h5 的 key 是 raw id，benchmark 的 GT instance id = `sem_id*1000 + raw_id + 1`。

（另查：官方 `arccos(|cos|)` 在 |cos|>1 时返回 nan，nan<th 为 False，**完美预测会被判失败**——但 4537 组比较中 0 次触发，不影响任何数字。）

## 匹配情况

- 场景 41/42（`a980334473` GT 为 0，跳过无损失）
- GT 实例 390（rotation 236 / translation 154），非空预测 3652（共 4100 queries）
- 贪心匹配（score 降序，IoU>0.5，GT 唯一占用）→ **275 个**
- recall 70.5%（rotation 76.3% / translation 61.7%）—— 这是 AP50=0.5711 的天花板来源

## 误差分布（275 对匹配成功的）

| 组 | n | axis 中位/均值 | p75/p90/p95 | <15° | <20° | <30° |
|---|---|---|---|---|---|---|
| ALL | 275 | 3.88° / 9.30° | 7.16 / 19.8 / 46.0 | 86.9% | 90.2% | 93.5% |
| rotation | 180 | **2.72°** / 8.65° | 5.59 / 19.8 / 47.4 | 88.3% | 90.0% | 93.9% |
| translation | 95 | 5.91° / 10.53° | 12.3 / 19.3 / 45.1 | 84.2% | 90.5% | 92.6% |

| 组 | n | origin 垂距 中位/均值 | p75/p90/p95 | <0.25m | <0.3m | <0.5m |
|---|---|---|---|---|---|---|
| rotation（计分） | 180 | **0.127** / 0.247 m | 0.266 / 0.563 / 0.813 | **72.8%** | 76.7% | 88.3% |
| translation（不计分） | 95 | 0.226 / 0.256 m | 0.357 / 0.444 / 0.597 | 56.8% | 68.4% | 92.6% |

**逐类 AP50 保留率 —— 最指方向的一张表：**

| 判据 | all | rotation | translation |
|---|---|---|---|
| M | 0.5711 | 0.7138 | 0.4283 |
| MA(15°) | 0.4419 | 0.5734 (80.3%) | 0.3104 (72.5%) |
| MO(0.25) | 0.4502 | 0.4720 (**66.1%**) | 0.4283 (100%，定义上跳过) |
| MAO_ST | 0.3214 | 0.3323 | 0.3104 |

→ **rotation 的最大漏点是 origin（只保留 66%），translation 的唯一漏点是 axis（72.5%）。**

## 发现 1：axis 塌陷到重力先验，横向铰链 100% 全错

轴角误差直方图（bins 0/2/5/10/15/20/30/45/60/75/90°）：

```
[66, 99, 58, 16, 9, 9, 3, 6, 1, 8, 0]
 └── 主峰 0–5° ──┘        中间近空      └ 85–90° 小峰
```

典型双峰。36 个失败样本中 18 个在 15–30°（近失），18 个 >30°，其中 **8 个在 85–90°（近乎正交）**。

```
匹配上的 rotation GT：168/180 (93.3%) 轴在竖直 ±15° 内
8 个横向铰链 → 全部失败，误差 [87.3, 89.2, 87.7, 90.0, 87.5, 87.1, 85.5, 49.0]°
              其中 7 个模型预测的都是 ≈(0,0,1)
```

**最扎心的一条：「永远输出重力方向」这个平凡基线在 rotation 轴上比模型还准**（93.3% vs 88.3%），win/tie/loss = **0 胜 / 9 负**。模型没救回任何一个先验答错的样本，只额外弄坏了 9 个。**模型的轴预测价值全部来自 translation**（那里先验 0%、模型 84.2%、80 胜 0 负）。

另：36 个 axis 失败里 **14 个预测轴其实与同场景另一个 GT 实例的轴吻合**（<15°），说明部分是 query 串了实例，不是纯回归噪声。

AP 层面：只 oracle 修掉 >30° 的 18 个 → 0.4419→**0.5078 (+0.066)**，即 6.5% 的样本贡献了 51% 的 axis 损失。

## 发现 2：origin 是系统性精度不足，不是长尾

rotation 直方图（bins 0/.05/.1/.15/.25/.3/.5/.75/1/2/∞ m）：

```
[43, 30, 27, 31, 7, 21, 10, 3, 6, 2]
```

中位误差 0.127 m 已是阈值的一半，49 个失败里 28 个落在 0.25–0.5 m，只有 2 个 >2 m。**整条分布就是偏宽。**

**没有值得修的全局平移偏差。** 表面上 world 系 mean Δz = −0.234 m 看着吓人，但分解后：竖直轴 rotation (n=168) 的**垂直分量中位数恰好 0.000** —— 这 −0.23 m 完全落在轴向上，度量根本看不见（GT origin 只是取了铰链线上更高的点）。全局最优平移 oracle 暴搜只能 +0.006。

**⚠️ 但训练损失在惩罚这个自由度**（见 [05-improvement.md](05-improvement.md) 的损失对齐一节）。

**真实的系统性是「向部件质心收缩」**：77.2% 的 rotation origin 被拉向预测部件质心，沿质心方向的中位分量 +0.089 m，而 GT origin 距质心中位 0.309 m —— **只回归了 29% 的距离**。不过模型仍远好于「直接用质心」基线（72.8% vs 27.8%）。

## 发现 3：小部件的轴更差，origin 与尺寸无关

| GT 点数 | n | 部件对角线中位 | axis 中位 | axis<15° | origin 中位(rot) | origin<0.25(rot) |
|---|---|---|---|---|---|---|
| <500 | 92 | 0.45 m | 5.59° | 0.826 | 0.067 (n=13) | 1.000 |
| 500–2k | 77 | 1.06 m | 5.16° | **0.792** | 0.147 (n=64) | 0.719 |
| 2k–8k | 106 | 2.28 m | 2.19° | **0.962** | 0.121 (n=103) | 0.699 |

- axis 误差与部件大小**强负相关**（Spearman −0.52 vs 点数）→ 小部件轴头分辨率不足
  （注意 <500 桶里 79/92 是 translation，混了「translation 更难」的因素）
- origin 误差与尺寸**几乎无关**（Spearman 0.006）→ **绝对尺度误差**，因此相对误差在小部件上被放大（err/部件对角线 中位 0.078，p90 0.342，max 2.90）

## 阈值放宽的量化

| axis 阈值 | 15° | 20° | 30° | 无约束 |
|---|---|---|---|---|
| AP50_axis | 0.4419 | 0.4689 | 0.5107 | 0.5711 |

| origin 阈值 | 0.25 m | 0.3 m | 0.5 m | 无约束 |
|---|---|---|---|---|
| AP50_origin | 0.4502 | 0.4778 | 0.5288 | 0.5711 |

联合：(15°,0.25)=0.3214 → (20°,0.3)=0.3830 → (30°,0.5)=0.4698

## 🎁 免训练后处理：val +0.0505

只用预测量，无 GT 泄漏，AP 用官方判据重算：

| 干预 | AP50 | AP50_axis | AP50_origin | AP50_axis_origin |
|---|---|---|---|---|
| baseline | 0.5711 | 0.4419 | 0.4502 | 0.3214 |
| **A.** rotation 轴距重力 <30° → 吸附为 z | 0.5711 | 0.4635 | 0.4502 | 0.3521 |
| **D.** origin 吸附到自身预测掩码最近点 | 0.5711 | 0.4419 | 0.4744 | 0.3471 |
| **A+D** | 0.5711 | 0.4635 | 0.4744 | **0.3718 (+0.0505, 相对 +15.7%)** |

rotation 分项：A+D 使 rot 的 MA 0.5734→0.6165，MO 0.4720→0.5205，MAO 0.3323→**0.4333**。

**AP50 完全不变**（后处理不动 mask）。

注意：
- **D 无超参、最安全**
- **A 的 30° 阈值是在 val 上选的**（30/45° 已是平台期），且会**把横向铰链的失败锁死**——但它们本来就 8/8 全错
- 反向操作（origin 沿质心方向外推）α≥0.3 就开始掉点，α=0.5 严重掉点 → **「收缩」存在但不能粗暴反演**

## 复现

```bash
source /data/challenge/USDNet/env.sh
cd /data/challenge/analysis
export PYTHONPATH=/data/challenge/USDNet:/data/challenge/analysis
python build_matches.py          # 校验：复现官方 0.5711/0.4419/0.4502/0.3214
python diagnose_arti_errors.py   # 分布 / 分桶 / 阈值扫描
python followup_bias.py          # 偏差方向 / 近正交个案 / 逐类 AP
python interventions.py          # 后处理干预
python make_summary.py           # -> SUMMARY_arti_diagnosis.json
```
