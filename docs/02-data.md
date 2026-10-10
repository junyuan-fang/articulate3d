# 数据：来源、验证结果、已知缺陷

## 数据构成

Challenge 需要两样东西，**不需要下载完整的 ScanNet++**：

| | 路径 | 说明 |
|---|---|---|
| Articulate3D 标注 | `/data/challenge/articulate3d_annotations/` | HuggingFace `INSAIT-Institute/Articulate3D` |
| ScanNet++ mesh | `/data/challenge/scannetpp_data/data/<scene>/scans/mesh_aligned_0.05.ply` | 只需这一个文件 |
| **预处理数据** | `/data/challenge/USDNet/data/processed/articulate3d_challenge_{mov,inter}/` | **训练/推理实际读这里** |

**优先用预处理数据**（USDNet 的 Google Drive）。官方明确说它与评测服务器**逐字节对齐**，自己预处理有点云顺序不一致的风险。

## Split

Articulate3D 把 ScanNet++ v1 的场景重新划分，**与 ScanNet++ 自己的 split 不同**：

```
train 195 / val 42 / test 42     场景 ID 列表：/data/challenge/data_download/articulate3d_splits.json
```

已交叉验证：从 challenge 页面抓取的列表与 HF 仓库里官方的 `train.txt`/`val.txt` **逐个 ID 完全一致**（test 官方未提供列表，只在页面上有）。

## 完整性验证结果

不是数文件个数，是实际读取校验：

| 检查项 | 结果 |
|---|---|
| 标注 JSON | 474 个全部 `json.load` 成功 |
| 预处理点云 `.npy` | 279 个，形状均为 (N, 13) |
| 关节 GT `.h5` | 237 个，1 个空（设计使然，见下） |
| `expand_dict/*.pkl` | 237 个全部 load 成功 |
| **database 引用的每个文件** | mov + inter 两套 × 4 个 yaml，**零缺失** |
| ScanNet++ mesh | 278 个全部通过 PLY 头校验 |
| 预处理 zip | CRC 全量校验通过 |

`.npy` 列含义：`0-2` xyz，`3-5` rgb，`6-8` normal，`9` 语义(0/1/2)，`10`/`12` 实例 id。

## 标注字段（比榜单用到的丰富得多）

```
articulations: pid, type, origin, axis,  base, rangeMin, rangeMax, motionStates, ref, needsReview
               └─ 榜单只评这4个 ─┘      └────── 公开了但榜单不评 ──────┘
```

共 **6349 条关节**（rotation 4687 / translation 1662），分布在 237 个场景。

**`base`（父部件层级）和 `rangeMin`/`rangeMax`（运动范围）是免费的额外监督信号，目前完全没用上。**

## 已知缺陷（三个，都已查清）

### 1. `280b83fcf3` 的原始 mesh 服务器已下架

```
scans/mesh_aligned_0.05.zip      404      ← 整个 scans/ 都没了
scans/pc_aligned.zip             404
iphone/rgb.mkv                   200      ← 但场景本身还在
```

该场景已从 ScanNet++ 的**所有 6 个 split 文件**中移除。`version=v2` 全部返回 406（下载脚本版本过旧），且**即使拿到 v2 的 mesh 也不能用**——Articulate3D 的 `_parts.json` 按 **mesh face 索引**存标注，v2 重新做过网格简化，face 顺序全变，标注会整体错位。

**影响：无。** 预处理数据里这个场景是全的（`train/280b83fcf3.npy` + `_articulation.h5` + pkl + instance_gt 都在），`train_database.yaml` 是完整的 195。只有当你要从原始 mesh 自己跑预处理时才会少这一个。

点云已从预处理数据导出到 `/data/challenge/derived/280b83fcf3/`（**是点云不是 mesh**，无面片且已体素下采样，同目录 README.txt 有说明）。

### 2. `a980334473` 的关节 GT 为空 —— 这是对的，不要"修"

`_articulation.h5` 是 800 字节的空文件（0 个 dataset），`.npy` 里 117117 个点的 sem/inst **全是 0**。但它的 `_artic.json` 里有 2 条 rotation。

**原因：GT 的生成规则是"从可交互部件出发，向上追溯到可动祖先"。** label 编码层级路径：

```
door frame.1.1
door.1.1.2_1                      ← 可动部件
handle base.1.1.2_1.3_1
handle.1.1.2_1.3_1.4_1            ← 可交互部件（叶节点）
```

`get_articulations()` 从叶节点（把手/旋钮/铰链）出发往上找可动祖先才生成一条 articulation。而 `a980334473` 的门和灯开关**都没有标任何子节点**，所以一条都生成不了。

我把这条规则逆向实现后在 237 个场景上验证，**230 个精确复现官方 GT 的实例集合**（剩下 7 个差 1–2 个实例，是我的 label 父子匹配写得粗糙）。

**所以不要往里注入标注** —— 评测服务器用同一套流水线产出 GT，改了本地 val 就和官方口径不一致了，本来想修 bug 会变成把评测搞错还不易察觉。

副作用：`trainer.py:560` 会跳过无 GT 的场景，所以 val 推理只产出 **41/42** 个场景。dev 提交因此缺 1 个场景，实测**服务器接受、不影响评分**。

### 3. ~~26% 的标注带 `needsReview`~~ —— 不存在这个问题

**这条曾经是我的误读，已作废，留在这里防止再犯。**

```
articulations 总数 6349；含 needsReview 字段 1667；缺字段 4682
字段取值分布: {False: 1667}      ← 没有一条是 true
```

我最初用 `Counter.update(a.keys())` 统计，数的是**字段出现次数**而不是取值，于是把 1667 读成了「1667 条待复核」。实际全部是 `false`。

**教训：统计 JSON 字段时，`update(d.keys())` 数的是 key 存在性，要看取值必须显式取 `d[k]`。**

## ⚠️ 用官方发布的数据没法自己跑预处理

```python
usda_files = [f for f in os.listdir(scene_folder) if f.endswith('.usda')]
self.usda_file = osp.join(self.scene_folder, usda_files[0])   # IndexError
```

预处理脚本需要每场景一个 `.usda` 文件解析部件层级树，而 **HF 发布的是 JSON，`.usda` 一个都没有**。README 第 189 行的 TODO 正是这件事，还没打勾：

> `- [ ] Merge data loader with json format to datapreprocessing`

**这不是我们的环境问题，所有参赛者都一样。** 实际后果：

- 只能用 Drive 上那份预处理数据，**改不了体素大小（0.01）等预处理阶段的设定**
- 想改预处理，得先自己写 `parts.json` → 层级树的解析器替换 `USDAParser`。规则我已逆向出来并验证过（见上），有现成基础

## 下载相关

`/data/challenge/data_download/` 下：

| | |
|---|---|
| `progress.sh` | 看进度 |
| `stop.sh` | 停止全量下载 |
| `supervise.sh` | 启动/恢复（机器重启后跑这个） |
| `challenge/download_meshes.py` | 只下 challenge 需要的 mesh（并行 + 容错） |

**踩过的坑：**

- **官方 `download_meshes.py` 遇到 404 会 `exit(1)` 中断整个下载** —— 按它的场景顺序会停在第 39 个（`280b83fcf3`），后面 240 个都下不到。我的版本记录后跳过。
- **`stop.sh` 曾经是坏的**：按绝对路径 pkill 匹配 `supervise.sh`，但启动时用的是相对路径 `./supervise.sh`，对不上 → 只杀掉 worker，supervisor 20 秒后又把它们拉起来。已改成按 basename 匹配，并**实测验证**（停止后等 25 秒确认没有残留进程重启）。
- TUM 服务器很慢很不稳，单流 ~0.3 MB/s，约 1/3 请求超时。所有下载都要重试 + 断点续传。
- ScanNet++ 下载 token 有效期 30 天，可在官网点 "Request token extension" 延期。


### 下载凭据

配置文件只保留占位符，下载脚本不再读取 YAML 中的 Token；分片生成器也会移除 `token` 字段。
直接运行 `python data_download/download_scannetpp.py <config.yml>` 时，可在终端隐藏输入 Token。
无人值守运行（包括 `run_download.py` 和 `supervise.sh`）必须先设置环境变量：

```bash
# 在 Bash / Zsh 中隐藏输入，避免把 Token 写入 shell 历史。
printf 'ScanNet++ token: '
read -r -s SCANNETPP_TOKEN
printf '\n'
export SCANNETPP_TOKEN
bash data_download/supervise.sh
unset SCANNETPP_TOKEN
```

下载进度、重试和错误消息会隐藏 Token。已外发的配置、旧日志、截图及 Git 历史中的凭据不会因当前文件修改而失效；需要向签发方撤销或更换旧 Token，再清理历史副本。


只下载 challenge mesh 的入口也使用相同的 `SCANNETPP_TOKEN` 环境变量或终端隐藏输入：

```bash
python data_download/challenge/download_meshes.py --data-root /data/challenge/scannetpp_data
```

该入口已移除 `--token` 参数，避免凭据进入命令行历史和进程参数。
