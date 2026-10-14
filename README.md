# Lingbot VLA / 天池 532514 训练代码

当前默认：**基础 BC 训练**（`v0.1.0-bc`）。三版共享当前训练实现，版本标签固定不同实验的默认配置。

|版本|初始化|epoch / update|RTC|图像裁剪 / 逐关节dropout|expert输出dropout|
|---|---|---|---|---|---|
|v0.1.0-bc|公开基模|10 / 77340|关闭|关闭 / 0|0|
|v0.2.0-rtc-continue|基础版完整HF模型|追加3 / 23202|开启|5% / 0.2|0|
|v0.3.0-rtc-fresh-dropout02|公开基模|10 / 77340|开启|5% / 0.2|0.2|

基础版使用历史颜色+状态噪声配置，状态噪声概率0.5；续训和重训版的数值状态噪声关闭。
逐关节dropout仅作用于12个臂关节的归一化输入，夹爪和动作标签保留。Expert dropout作用于attention输出和FFN/MoE输出；整份状态特征dropout默认0。

## 环境

Linux、Python **3.11**、PyTorch **2.8.0+cu128**、torchvision **0.23.0+cu128**、Triton **3.4.0**、Transformers **4.57.3**。还需要FFmpeg、C++编译器和支持CUDA 12.8的NVIDIA驱动。其余依赖固定在 `requirements.txt`。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

也可执行 `docker build -t lingbot-tianchi:cu128 .` 构建源码环境；Docker挂载与DLC步骤见 [部署说明](docs/DOCKER_DLC.md)。DLC已有相同环境时可直接使用本仓库。训练不自动修改依赖。

## 数据与基模

代码包含原生Lingbot源码和Qwen处理器的配置/词表；运行资产通过下列两种方式准备。

1. 使用已有工作目录的统计文件，保持原checkpoint的归一化及split哈希：

```bash
python scripts/import_runtime_assets.py --from-root /path/to/existing/project
```

此命令只复制三个小文件：split、增强统计和原生归一化。它们位于Git忽略的运行目录。

2. 首次使用官方2500条clean数据时，重新生成统计：

```bash
python -m augmentation.prepare --dataset /path/to/RoboTwin_lerobot_v21 --seed 532514 --validation-fraction 0.1
python -m training.prepare_norm
```

从公开 `robbyant/lingbot-vla-v2-6b` 权重开始；模型/data固定来源见 `manifests/download_manifest.json`。
完整轨迹分割为2250条训练、250条验证。每次读取重新随机增强，一轮仍是494969个训练帧，不额外生成轨迹文件。
RTC续训应复用基础训练的split和norm，使用完整HF模型初始化新的optimizer/scheduler。

## 配置与前台训练

选择配方并生成本机路径配置（所有路径也可填共享盘的绝对路径）：

```bash
python scripts/configure.py --profile bc_baseline \
  --base-model /path/to/base_or_finetuned_hf_model \
  --dataset /path/to/RoboTwin_lerobot_v21 \
  --output /path/to/persistent/outputs/bc_baseline
python -m training.preflight --cpu-only --config configs/runtime.json
NUM_GPUS=4 bash scripts/train.sh
```

启动日志直接输出到stdout/stderr。DLC应将模型、数据和输出配置为已挂载的共享盘路径；代码目录可以位于容器本地盘。
默认4卡，每卡micro batch 8，梯度累积2，全局batch 64；1–8卡均支持，实际batch和累积次数打印在日志中。
预测长度50；RTC训练最大前缀延迟12、30%无前缀，部署建议每25条动作更新。

```bash
NUM_GPUS=4 bash scripts/train.sh --epochs 5
NUM_GPUS=4 bash scripts/train.sh --micro-batch 4
NUM_GPUS=4 bash scripts/train.sh --expert-dropout 0.2
NUM_GPUS=4 bash scripts/train.sh --resume /path/to/outputs/step_008000
```

同一实验的resume需保持RTC、dropout、batch、split、统计等设置一致。
基础版/重训版每8000步保存、保留10个；续训版每3000步保存、保留8个。结束时额外保存最终断点并导出 `step_XXXXXX/hf_ckpt`，中间断点包含可恢复训练的权重、optimizer和RNG。

## 检查和来源

```bash
python -m unittest augmentation.test_clean_augment augmentation.test_joint_dropout augmentation.test_image_crop training.test_distributed training.test_checkpoint training.test_rtc
python -m simulation.check_rtc
```

上述CPU检查不需要训练数据和模型权重。真实模型/meta架构检查可在准备资产后运行 `python -m training.check_action_dropout --config configs/runtime.json`（expert dropout应开启）。

项目结构、算法接入与来源见 [代码导览](docs/CODE_MAP.md)、[实验记录](docs/EXPERIMENTS.md)、[第三方来源](THIRD_PARTY_NOTICES.md)、[版本说明](CHANGELOG.md)。正式成功率仍需闭环仿真验证。
