# 版本标签

三版共享当前训练实现，主要差异是初始化源、训练预算、增强、RTC与dropout参数。

- `v0.1.0-bc`：公开基模，10epoch，历史颜色+数值状态噪声，无RTC/逐关节dropout/expert dropout。
- `v0.2.0-rtc-continue`：基础版完整HF模型初始化，重新建optimizer，追加3epoch；RTC硬前缀、裁剪、逐关节dropout，数值状态噪声关闭。
- `v0.3.0-rtc-fresh-dropout02`：公开基模，10epoch；沿用RTC增强，attention输出与FFN/MoE输出dropout各0.2。

导出整理：相对路径配方、独立runtime配置、前台1–8卡启动、运行资产导入、CPU测试临时统计fixture、环境/源码说明。
