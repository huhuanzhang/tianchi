# 代码导览

|位置|作用|
|---|---|
|training/configs|三版可审阅配方；default.json对应当前标签|
|scripts/configure.py|生成Git忽略的本机runtime路径配置|
|scripts/train.sh|前台启动preflight与torchrun，NUM_GPUS选择卡数|
|training/native_model.py|原生模型装配、完整权重检查、冻结VLM与动作模块FP32|
|training/train_server.py|DDP、梯度累积、训练预算、验证、保存、resume与保留策略|
|training/clean_dataset.py|clean映射、原生55维空间、在线增强、尾部padding|
|augmentation|颜色、裁剪、逐关节dropout与仅训练集统计|
|training/rtc.py|训练期硬前缀、后缀监督、反向时间Euler采样|
|simulation/rtc_guidance.py|推理期VJP guidance算法参考|
|simulation/rtc_moe.py|可微PyTorch MoE参考与分析检查|
|third_party/lingbot-vla-v2|带原生接入修改的Lingbot源码|

算法接入：`modeling_lingbot_vla_v2.py`、`modeling_lingbot_vla.py`、`qwen2_action_expert.py`。
Expert dropout仅训练启用，eval关闭；标签与动作空间定义保持完整。此源码包提供训练和RTC核心，RoboTwin场景资产与任务评测工程另行准备。
