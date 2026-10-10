# 环境：在 RTX 5090 (sm_120) 上跑 Mask3D 系代码

## 结论先行

**不用编 Docker。**

基础环境为 **python 3.10 + torch 2.7.1+cu128 + MinkowskiEngine 0.5.8 (alpsaur/MinkowskiEngine@8f76f27)**（针对 sm_120 编译），实测在 5090 上 GPU 稀疏卷积正常。在此基础上再装 USDNet 依赖即可。

```bash
source /data/challenge/USDNet/env.sh    # 激活 conda env `usdnet`
```

验证：
```python
import torch, MinkowskiEngine as ME
torch.cuda.get_arch_list()        # 含 'sm_120', 'compute_120'
torch.cuda.get_device_capability(0)   # (12, 0)
```

## 为什么官方 Dockerfile 在 RTX 5090 上不可用

```dockerfile
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9"        # 不含 sm_120
RUN pip install torch==2.1.0 ... --index-url .../cu121  # cu121 完全不支持 Blackwell
```

RTX 5090 是 sm_120 (Blackwell)，需要 CUDA 12.8+ 和 torch ≥2.7.0（2.7.0 是第一个提供 sm_120 预编译 wheel 的稳定版）。

## 环境构建步骤（复现用）

```bash
# 前提：conda env `usdnet` 已装好 python 3.10 + torch 2.7.1+cu128 + MinkowskiEngine 0.5.8 (alpsaur/MinkowskiEngine@8f76f27)
#       （完整版本见 env/usdnet_pip_freeze.txt）

P=$HOME/.conda/envs/usdnet/bin
# 必须先降 pip，见下方坑 #1
$P/python -m pip install "pip==23.1" "setuptools<70"
$P/pip install "omegaconf==2.0.6" "hydra-core==1.0.5" "antlr4-python3-runtime==4.8" \
               "pytorch-lightning==1.7.2" "torchmetrics==0.9.3" \
               natsort python-dotenv volumentations "albumentations==1.2.1" \
               pyviz3d usd-core peft "pillow==9.5.0"

# detectron2 需要 CUDA_HOME 指对，见坑 #2
export CUDA_HOME=$HOME/.conda/envs/usdnet
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=16
$P/pip install --no-build-isolation --no-deps \
  'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf'
```

关键版本组合（实测可用）：

| | |
|---|---|
| torch | 2.7.1+cu128 |
| MinkowskiEngine | 0.5.8 |
| pytorch-lightning | **1.7.2**（Mask3D 用的是 PL 1.x API，2.x 会崩） |
| hydra-core / omegaconf | **1.0.5 / 2.0.6**（Mask3D 硬依赖这个古董组合） |
| Pillow | **9.5.0**（见坑 #3） |
| detectron2 | 0.6 |

## 踩过的坑

### 坑 1：`omegaconf==2.0.6` 装不上

```
ERROR: Could not find a version that satisfies the requirement omegaconf==2.0.6
  Ignoring version 2.0.6 ... has invalid metadata: PyYAML (>=5.1.*)
  Please use pip<24.1 if you need to use this version.
```

它的元数据在 pip ≥24.1 下不合法。**先 `pip install "pip==23.1"`**。官方 Dockerfile 也是这么做的，容易被忽略。

### 坑 2：编译扩展报 CUDA 版本不匹配

```
RuntimeError: The detected CUDA version (11.5) mismatches the version
that was used to compile PyTorch (12.8).
```

系统里有多个 CUDA：系统 PATH 里的 nvcc 是 11.5，`/usr/local/cuda` 指向 13.2，**而正确的 nvcc 12.8.61 在 conda env 内部**。`env.sh` 里已经把 `CUDA_HOME` 指过去了。任何时候重编 CUDA 扩展都要先 source 它。

### 坑 3：`module 'PIL.Image' has no attribute 'LINEAR'`

Pillow ≥10 删了 `Image.LINEAR`，detectron2 0.6 还在用。降到 `pillow==9.5.0`。

降级后 pip 会报一堆冲突，**都是良性的**：
```
detectron2 0.6 requires hydra-core>=1.1, but you have hydra-core 1.0.5
detectron2 0.6 requires omegaconf>=2.1, but you have omegaconf 2.0.6
scikit-image 0.25.2 requires pillow>=10.1, but you have pillow 9.5.0
```
USDNet 只用了 detectron2 里 `point_rend.point_features.point_sample` 和 `utils.comm.get_world_size` 两个纯 Python 函数，不碰它的配置系统。

### 坑 4：`torch.load` 加载不了官方 checkpoint

```
_pickle.UnpicklingError: Weights only load failed...
```

torch 2.6 把 `torch.load` 的 `weights_only` 默认值从 `False` 改成 `True`。已改 `utils/utils.py` 里 7 处，加 `weights_only=False`。与上游的差异见 `usdnet_patch/usdnet_0ba303d.patch`。

### 坑 5：`set -u` 下 source env.sh 会退出

```
~cuda-nvcc_activate.sh: line 41: NVCC_PREPEND_FLAGS: unbound variable
```

conda 的 cuda-nvcc 激活钩子无条件追加这个变量。`env.sh` 里已给它兜了默认值。

### 坑 6：hydra 1.0.5 的异常打印在 Python 3.10 下自己会崩

```
TypeError: print_exception() got an unexpected keyword argument 'etype'
```

**正常运行不受影响**，但一旦出错，真实报错会被这个二次异常盖住。**排查问题时记得往上翻日志找第一个 traceback。**

## 对仓库做的修改

| 文件 | 改动 | 原因 |
|---|---|---|
| `utils/utils.py` | 7 处 `torch.load` 加 `weights_only=False` | torch 2.6 破坏性变更（备份 `.orig`） |
| `env.sh` | 新增 | 环境设置 |
| `eval_val.py` | 新增 | 仓库只暴露 `fit`/`test`，缺 `validate` 入口，见 [04-diagnosis.md](04-diagnosis.md) |
| `scripts/infer_mov_local.sh` | 新增 | 路径填好、wandb 离线 |
| `scripts/infer_mov_val.sh` | 新增 | val 评测 |

## 仓库内置但未使用的能力

`trainer.py` 里已经接好了两个"保住分割、只调关节"的机制：

```python
# trainer.py:84    PEFT LoRA
general.LORA.{USE_LORA, R, ALPHA, DROPOUT, TARGET_MODULES}
# trainer.py:113   骨干冻结
general.freeze_backbone
```

见 [05-improvement.md](05-improvement.md)。


## 发布配置的渠道迁移

仓库内的 Dockerfile 已改用 Miniforge，移除自动接受 Anaconda 条款的命令。
两份 Conda YAML 使用 conda-forge（当前 CUDA 导出另保留 NVIDIA 渠道），并以
`nodefaults` 禁止追加默认渠道。旧版 YAML 的 Anaconda 专属构建编号及元包已移除。
这两份文件是迁移后的版本规格，不再是原环境的逐字导出；尚未在 Linux/CUDA 上
重新求解或构建，不能视为经过验证的锁文件。既有环境的测试记录不代表迁移后构建已通过。

渠道选择参考：https://github.com/conda-forge/miniforge 。各依赖本身的许可证仍分别适用。
