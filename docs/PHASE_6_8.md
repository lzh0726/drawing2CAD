# 数据整理、基线及 LoRA 实验

工作目录：`/home/luzhuohong/drawing2CAD`。

## 边界

原始 ZIP 保持不变。Ortho2CAD 数据来源许可证尚未明确，所有记录保留
`license_status: unresolved`，不能据此声称获准公开再分发或商业使用。
本次属于自动检查的研究候选集，`human_reviewed` 始终为 false；不修改已有人工审核导出器。
还没有启动 LoRA 训练。

## 数据流程

`tools/prepare_pilot.py` 使用固定种子 42，从官方 DeepCAD 100 个测试项中固定取 50 个。
全部 100 个官方测试项的代码指纹、参数无关 AST 模板都被排除出训练候选。
训练、验证按模板哈希划分，目标分别为 1000、100；过滤代码重复、几何描述签名重复、
执行失败以及保守包围盒与精确包围盒相差超过 0.5% 的候选。
几何描述签名是一项保守去重启发式，并不是完整的几何等价证明。
没有全面解决所有原始图纸标注问题，也未覆盖缺失的旋转、放样等操作。

只向标签末尾增加 `result = solid`，不更改模型形状或整体缩放。
官方 STEP 导入遵循文件单位；所有结果使用毫米。
官方无效或多实体真值保留问题记录，不得计为模型几何错误。

输出：`datasets/pilot-v1/manifest.jsonl`、images、codes、steps 和 preparation-report.json。
首次生成完成后拒绝覆盖已有 manifest，避免静默改变测试集。

`tools/export_pilot_sft.py` 使用实际 LLaMA-Factory 模板编码完整图文序列，
超过 4096 token 的样本排除并记录，不截断标签。图片像素上限 262144。
因此最终训练条数可能少于候选条数，应以 export-report.json 为准。
输出到 `datasets/sft-pilot-v1/`，使用服务器绝对图片路径。

## 基线

`tools/run_pilot_baseline.py` 仅选择显存占用小于 256 MiB、GPU 利用率为零且没有计算进程的显卡。
共享机器检查存在时间竞争；正式长期任务宜通过管理员的调度系统分配显卡。
原始 Qwen3-VL-8B-Instruct，贪心解码，最多生成 2048 token，无修复。
输出保存在 `experiments/baseline-pilot-v1/`，每个样本独立落盘，可继续未完成项。

记录原坐标 IoU、仅平移对齐 IoU、逐轴绝对/相对尺寸误差、有效实体率、耗时、峰值显存。
不做尺度对齐。针对无效官方真值和比较计算失败，需要单独汇总，不能混作预测错误。
外观与关键尺寸的人工审核仍需实际进行；这些自动指标不能替代该审核。

## 执行隔离

离线 CAD 使用 Bubblewrap 的无网络、只读文件系统、独立 PID/设备空间，
配合 AST 策略、CPU/内存/输出限制。仅挂载依赖环境和可信运行入口。
不改变在线服务现有 Docker 执行边界。训练环境与 CAD 环境独立。

## 训练配置

`configs/lora-pilot-v1.yaml`：LoRA rank 16、alpha 32、batch 1、梯度累积 16、
2 epochs、学习率 1e-4、BF16、SDPA；冻结视觉塔和多模态投影层。
CPU 参数解析只验证配置字段；它不代表已完成 GPU 训练或显存验证。
先确认数据审核状态、完整基线和数据导出报告，再决定是否进入训练。
停止本任务自己的推理进程并确认 GPU 空闲后，训练命令为：

```bash
cd /home/luzhuohong/drawing2CAD
CUDA_VISIBLE_DEVICES=获准使用的空闲编号 \
  /home/luzhuohong/miniconda3/envs/drawing2cad-train/bin/llamafactory-cli \
  train configs/lora-pilot-v1.yaml
```

这条命令是后续操作说明，本阶段不会自动执行。
