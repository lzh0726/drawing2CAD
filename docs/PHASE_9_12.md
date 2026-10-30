# 第 9–12 步：当前准备状态

2026-10-25 检查：四张 A6000 均有 VLLM 计算进程，每张约 44 GB 显存、100% 利用率。
未启动试训或正式训练，未停止其他人的进程。此状态是检查时的快照。

## 9. 20 步试训

配置 `configs/smoke-pilot-v1.yaml` 已建立：真实训练数据最多 32 条、20 步、梯度累积 1。
命令在项目根目录执行：

```bash
/home/luzhuohong/miniconda3/envs/drawing2cad-train/bin/python tools/launch_training.py smoke
```

启动器两次核对 GPU 利用率、显存和计算进程；没有空闲显卡则退出，不排队、不抢占。
项目文件锁只能避免本项目重复启动，无法代替共享服务器的调度系统。
仍需实际确认 loss 有限、前后向与参数更新完成、仅预期 LoRA 参数可训练、adapter 输出完整、峰值显存。

## 10. 正式训练

配置 `configs/train-pilot-v1.yaml` 已建立。每 25 步保存并验证，按验证 loss 加载最优 checkpoint，
避免用测试集挑选权重。其余采用第 8 步参数。

```bash
/home/luzhuohong/miniconda3/envs/drawing2cad-train/bin/python tools/launch_training.py train
```

启动前要求试训完成 20 步、loss 有限且 adapter 已保存；仍必须核对试训日志。
不覆盖既有实验目录。试训和训练不并行运行。

## 11. CPU 合并

配置 `configs/export-pilot-v1.yaml` 已建立，导出到 `models/drawing2cad-v1`。
原模型约 17.53 GB，检查时磁盘仅约 24.73 GB 可用，余量较紧。
合并前重新核对磁盘与可用内存，保留原模型与 adapter，不自动删除文件。

```bash
/home/luzhuohong/miniconda3/envs/drawing2cad-train/bin/python tools/export_checked.py
```

配置和处理器可加载，只能证明导出元数据可用；还要实际加载权重并生成。

## 12. 同测试集比较与页面接入

尚未运行。A：已完成的原始模型零修复；B：微调模型零修复；C：微调模型最多修复两次。
三者必须采用相同图纸、像素上限、提示词和评估口径；无效标准 STEP 单独记录。
不得用测试集选择 checkpoint。

服务器目前没有 Docker，在线服务原有 Docker 执行条件尚不满足。
离线 Bubblewrap 验证器不等于在线接口已经部署成功；不能绕过隔离直接执行模型代码。
需要解决在线隔离运行条件后，再验证认证、SSH 隧道、上传图纸、反馈修复、返回 STEP 和页面预览。

许可证仍未明确、human_reviewed 仍为 false；不将自动检查描述为逐条人工审核。
