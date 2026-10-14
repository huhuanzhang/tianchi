# Docker 与 DLC

## Docker

先按README准备split和统计文件，再按容器路径生成runtime配置：

```bash
python scripts/configure.py --profile rtc_fresh \
  --base-model /assets/models/lingbot-vla-v2-6b \
  --dataset /assets/data/RoboTwin_lerobot_v21 --output /outputs/rtc_fresh
docker build -t lingbot-tianchi:cu128 .
docker run --rm --gpus all --ipc=host \
  -e NUM_GPUS=4 -e LINGBOT_CONFIG=/workspace/configs/runtime.json \
  -v "$PWD/configs/runtime.json:/workspace/configs/runtime.json:ro" \
  -v "$PWD/augmentation/artifacts:/workspace/augmentation/artifacts:ro" \
  -v "$PWD/training/artifacts:/workspace/training/artifacts:ro" \
  -v /path/to/assets:/assets:ro -v /path/to/outputs:/outputs \
  lingbot-tianchi:cu128
```

选择RTC续训时，将profile换为rtc_continue，base-model指向基础版完整HF checkpoint；使用相同split和norm。
runtime配置、统计目录与训练资产在运行时挂载。Docker构建上下文只包含版本化源码及处理器元数据。

## DLC

已有Python 3.11 / Torch 2.8 cu128环境时，在代码目录准备runtime配置，并将base-model、dataset和output设置为已挂载共享盘的绝对路径。
运行 `NUM_GPUS=4 bash scripts/train.sh`，日志直接显示在DLC控制台。
准备统计可使用 `scripts/import_runtime_assets.py` 复用原工程的小型统计文件，或按README重新计算。
运行前可执行 `python -m training.preflight --cpu-only --config configs/runtime.json` 检查路径及真实输入；CPU预检不启动正式训练。
