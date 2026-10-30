# 工具目录说明

保留文件用途如下。无其他文件引用的命令行入口和测试也可能有用，不能据此直接删除。

| 文件/目录 | 用途 |
|---|---|
| LLaMA-Factory/ | 第三方训练框架，A2/A3 及后续 SFT 训练依赖。 |
| cad-runtime/ | 隔离执行环境的受信任 worker、策略及诊断模块；server.research_sandbox 和 A2/A3 评估脚本依赖。 |
| prepare_pilot.py | 构建研究数据集：分组划分、去重及隔离 CAD 校验；也供数据准备测试使用。 |
| export_pilot_sft.py | 把筛选后的研究样本导出为图文 SFT 格式，保留原有审核状态。 |
| launch_training.py | 启动独立的 smoke/train/export 阶段，并检查 GPU；A1 评估工具也导入其 GPU 检查函数。 |
| export_checked.py | 在 CPU 上校验并合并基座与 LoRA，导出完整模型。 |
| run_pilot_baseline.py | 使用原始基座对固定测试集生成预测，建立基线。 |
| evaluate_finetuned.py | 使用已合并的微调模型评估固定测试集。 |
| reevaluate_entity_rate.py | 通过已部署服务重新评估固定测试集的实体生成率。 |
| audit_step_shapes.py | 在隔离环境内检查 STEP 的尺寸、体积、实体有效性与网格拓扑。 |
| run_success_geometry_audit.py | 对 A1 成功生成的 STEP 做尺寸与几何审计。 |
| start_online.py | 启动按需加载 GPU 模型的 CPU 接口服务。 |
| test_demand_model.py | 按需加载、释放模型等逻辑的回归测试。 |
| test_gpu_sharing.py | GPU 共享准入与资源规则的回归测试。 |
| test_model_input.py | 用 CPU 生成桩检查多模态输入预处理兼容性。 |
| test_online_executor.py | 用模型桩和真实隔离执行器检查 API 生成、导出和错误处理。 |
| test_pilot.py | 检查数据分组、代码去重和隔离 CAD 校验。 |
| test_reevaluate_entity_rate.py | 检查实体生成率统计及失败分类。 |

## 本次移出的已完成工具

这些脚本已从 tools 移出，并保留 SHA256 校验备份，便于历史复现时恢复。

- audit_ortho_dataset.py：首次 Ortho2CAD 数据压缩包静态审计，结果已归档。
- audit_rotations.py：旧旋转误差诊断辅助程序，诊断已完成。
- check_audit_step_geometry.py：首次数据审计的 STEP 检查辅助脚本，仅由旧审计汇总脚本调用。
- continue_after_smoke.py：首次 smoke 测试后的训练、导出、评估自动衔接，流程已完成。
- finalize_dataset_audit.py：汇总首次数据审计，结果已归档。
- finish_pilot.py：衔接首次 1,000 条数据准备与 SFT 导出，流程已完成。
- prepare_training_configs.py：一次性生成初始 smoke、训练和导出配置；配置文件已保留。
- run_rotation_audit.py：对固定历史样本补做旋转诊断，结果已归入 A1。

归档绝对路径：/home/luzhuohong/drawing2CAD/experiment-archives/tools-cleanup-20261029-160520

仅清理 tools 顶层的一次性脚本和 __pycache__，保留训练框架、隔离运行时、数据处理、模型导出、评估及测试。