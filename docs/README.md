# Articulate3D Challenge — 工作记录

> 本目录是赛期工作日志，各文件中的成绩与排名按当时日期记录，后来多有变化；最终成绩以仓库根目录 README 为准（dev 两榜第一，test 两榜第二）。

OpenSUN3D workshop Track 3（ECCV 2026）。目标：3D 室内场景中分割可动部件并预测其关节参数。

## 当前状态（2026-08-04）

| | |
|---|---|
| Baseline 复现 | ✅ test `AP50_axis_origin = 0.2640`，与榜单官方 baseline 0.26 一致 |
| **最终 test 成绩** | **0.3384512**（id=24，2026-08-11）——**排名指标榜首** |
| 榜单位置 | **0.33845（我们）** / linxii 0.33601 / ZZZ 0.26343 / ShinNam! 0.24670 |
| 四项对榜首 | AP50 **0.43697**>0.43524 ✓、origin **0.36015**>0.35472 ✓、axis 0.41008<**0.41346** ✗（差 0.0034）、joint **0.33845**>0.33601 ✓ |
| 唯一还输的一项 | **axis**。关节侧后处理已到顶（[07 §5](07-sota-options.md)），补它只能靠训练 |
| 已提交 | mov: id=11(dev) 12/15/17/24(test)；inter: id=13(dev) 14(test) |
| 截止 | **2026-08-26 07:59:59** |
| 训练中 | `v3_origin_bnfix`：只训 origin 分支 + BN 冻结，150 epoch，约 26h。**第一轮归因干净的实验** |
| 训练现状 | v1/v2 三轮的"提升"全部是 BatchNorm 漂移假象，已推翻，见 [06 §9](06-finetune-design.md) |

## 文档索引

| 文件 | 内容 |
|---|---|
| [01-environment.md](01-environment.md) | RTX 5090 上跑通 Mask3D 系代码的完整方案 + 踩过的坑 |
| [02-data.md](02-data.md) | 数据来源、完整性验证结果、已知缺陷 |
| [04-diagnosis.md](04-diagnosis.md) | 关节参数误差诊断（axis / origin 病因完全不同） |
| [05-improvement.md](05-improvement.md) | 改进方向，含文献 SOTA 做法与优先级 |
| [06-finetune-design.md](06-finetune-design.md) | 关节头 v2 的设计、消融与失败记录 |

## 目录布局

```
/data/challenge/
├── docs/                      本文档
├── USDNet/                    baseline 代码（git repo，含我的修改）
│   ├── env.sh                 环境（先 source 它）
│   ├── eval_val.py            我加的 validate 入口（仓库原本没有）
│   ├── scripts/infer_mov_local.sh   test 推理 → 提交 zip
│   ├── scripts/infer_mov_val.sh     val 评测 → AP 指标
│   ├── data/processed/        预处理数据（训练/推理实际读这里）
│   └── results/               推理输出与提交文件
├── analysis/                  误差诊断脚本与结果
├── articulate3d_annotations/  HF 原始标注 JSON
├── scannetpp_data/            ScanNet++（mesh + 全量下载中）
├── ckpts/                     checkpoints
├── data_download/             下载脚本与日志
├── derived/                   派生数据
└── evalai_token.txt           提交凭证（600 权限，勿入 git）
```

## 最重要的五条经验

1. **本地 val 评测与服务器逐位精确一致**（已验证四个指标全部对上）。调模型用 `scripts/infer_mov_val.sh` 即可看分，**不要浪费 test 的 9 次配额**。
2. **效果不差，是任务难。** 论文报告 AP50 41.8% / +Axis 25.0% / +Origin 34.6%，我们 test 是 43.70 / 38.01 / 31.60 —— 分割和轴都超过论文。ICCV25 榜首 0.41 也只比 baseline 高一档。
3. **分割不是瓶颈。** 我们 AP50 (0.4370) 是榜上最高，差距全在关节参数保留率（我们 60.4% vs linxii 77.2%）。不要去换分割骨干。
4. **val 涨的量，到 test 只兑现约 1/3**（实测 val +0.061 → test +0.019）。判断某个改动值不值得占用 test 配额时按这个折算。另外**推理是不确定的**（`seed: null`），同代码重跑波动 ±0.002 —— 小于这个量级的差异是噪声。
5. **「冻结」有歧义：冻参数 ≠ 冻模块行为。** `requires_grad=False` 管不住 BatchNorm 的 running buffer，这曾让整整三轮训练的结论作废（详见 [06 §9](06-finetune-design.md)）。声称"某部分被冻住"之前，**端到端比对两个 checkpoint 的 state_dict**，别信语义。

## 这份文档里被推翻过的结论

写下来是为了避免重蹈覆辙——每一条当时都看起来很有说服力：

| 曾经的结论 | 实际 | 根因 |
|---|---|---|
| 26% 标注带 `needsReview` | 一条都没有 | 统计的是字段出现次数而非取值 |
| 轴未归一化不影响评分 | 影响榜单指标，归一化白拿 +0.018 | 只读了 `MO` 判据没读 `MAO_ST` |
| `dist_pred_axis` 是多余项 | 评测里有对应项 | 同上 |
| origin 吸附到第一主方向端点 | 掉 0.19，应为垂直于轴 | 从 oracle 外推，未端到端验证 |
| 冻结后 AP50 不可能变 | BN buffer 会漂 | 混淆了参数与 buffer |
| origin attention 有效 (+0.015) | 全是 BN 漂移假象 | 上一条导致的归因污染 |
| 平面部件的滑动方向 = 面板法向 | 推拉门沿面内滑，恰好差 90° | 只想到抽屉一种物理形式 |
| 横向铰链"信息不存在，学不出来" | **方向一直都在**：面板面内水平方向，8 个里 6 个 <15°、中位 4.8°（[07 §4.7](07-sota-options.md)） | 探针测的是"骨干特征能否分类"，没测"几何里有没有这个方向" |
| 横向铰链的病根是 softmax 先验墙 | 病根是**锚点集**：即使选择器完美，现有候选也差 35°，天花板仅 +0.6pp；换成几何锚点是 +3.3pp | 只诊断了"分类器动不动得了"，没问"选对了又能得到什么" |
| superpoint「数据已备」 | 预处理里 segments 那列**全是 0**，从没算过 | 看了配置项存在就当数据存在 |
| 探针证明「横向铰链信息不存在」 | 那次探针只有 **7 个正样本**，统计上不足以证伪。137 样本的新探针结论相同，但这才是有效证据 | 把低功效实验的阴性结果当成了定论，并据此引用了两天 |
| origin 的 mask 天花板 100% ⇒ 后处理够用 | 天花板用了 GT，**oracle 高 ≠ 可达**；随后 4 个方法全部选不中 | 把"存在性"当成了"可达性" |
| 关节侧还差在"规则不够好" | 三个因子全部卡在**选择**而非表示：候选集里有答案，没有信号能选出来 | 一直在改规则，没问"凭什么选" |
| 匹配实例保留率涨 = AP 涨 | 保留率 91.6→96.8% 而 AP50_axis 掉 0.010 | 保留率只看高分匹配，AP 积分整条 PR 曲线 |
| 后处理不碰 mask，所以 AP50 恒定 | 换 checkpoint 后 AP50 0.4370→0.4366 | mask 确实没变，但 v7 重训改了 `pred_classes`/`pred_scores`，AP 依赖打分排序 |
