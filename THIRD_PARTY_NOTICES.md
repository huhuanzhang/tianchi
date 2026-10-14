# 第三方来源

- 原生模型代码：Robbyant/lingbot-vla-v2，Apache-2.0；保留各目录原始LICENSE与版权头。
- Qwen3-VL处理器/词表元数据来自官方Qwen3-VL-4B-Instruct配置快照。
- RTC inference-time参考：Real-Time Execution of Action Chunking Flow Policies，https://arxiv.org/abs/2506.07339 。
- 训练期硬前缀参考与本工程时间约定见 `training/rtc.py`；t=1为噪声、t=0为动作。
- GR00T N1.7的action head dropout参数参考：https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/configs/model/gr00t_n1d7.py 。本仓库使用Lingbot expert残差分支实现。

源码快照是上游及本工程修改后的组合，固定上游commit与修改文件哈希见 `SOURCE_PROVENANCE.json`。
模型权重和数据的来源/版本见 `manifests/download_manifest.json`，运行资产另行准备。
