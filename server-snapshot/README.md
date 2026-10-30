# 服务器源码与实验快照

采集日期：2026-10-30。通过只读 SSH 采集，不启动训练、评估或 GPU 服务，不改变服务器文件。

## 目录对应

| 仓库位置 | 服务器来源或用途 |
|---|---|
| `project/` | `/home/luzhuohong/drawing2CAD` 中的项目源码与配置 |
| `project/server/` | 服务器当前推理、隔离执行与几何检查实现 |
| `project/tools/` | 数据整理、训练启动、评估及测试工具 |
| `project/experiments/a2-10k-r16/` | A2 训练配置、启动器、资源监控代码 |
| `project/experiments/a3-10k-r16-projector/` | A3 投影层训练配置、训练入口及 CPU 审计工具 |
| `project/experiments/a1-1k-r16/` | A1 评估协议、汇总与几何诊断工具 |
| `experiment-runs/a2-test-20261028-v1/` | A2 测试评估脚本、汇总及逐例指标 |
| `experiment-runs/a3-test-20261029-v1/` | A3 测试评估脚本、汇总及逐例指标 |
| `project/experiment-archives/` | 已归档的历史工具、诊断脚本与实现版本 |

A4 的 30,000 条训练集准备脚本位于：

`project/experiment-archives/cleanup-20261029-153243/select-30k-20261028/select_30k.py`

同目录保留旧版本用于追溯；实际无长度过滤的最终脚本为 `select_30k.py`，不是 `before-no-length-filter.py`。

## 复用方式与版本边界

- 根目录为本地工作台源码，`project/` 为服务器版本。7 个同名源码文件内容不同，因此保留独立快照，不自动合并。
- 在新环境中使用服务器代码时，应将 `project/` 内容放到独立项目目录，并重新安装依赖、准备模型与数据。这里不是包含运行环境的可直接启动镜像。
- 脚本中的 `/home/luzhuohong/...` 和 `/data/luzhuohong/...` 是原实验路径，复现前需要对应部署或调整；归档后的目录位置也可能与脚本内原路径不同。
- 监控、启动、部署脚本只是源码备份。不要为了查看代码直接运行它们；执行可能启动训练或修改部署。
- LLaMA-Factory 第三方仓库没有重复打包，服务器版本为 `95ac3f2373b82662c1bd855c284d3379e6a763d3`，上游为 <https://github.com/hiyouga/LLaMA-Factory>。本次只读检查确认已跟踪文件无本地改动；未打包第三方工作区的未跟踪文件。
- A2 指标应使用 `summary-serial.json`，`summary.json` 为此前阶段记录。所有报告均为历史实验记录，不是本次上传重新运行得到的结果。

## 包含与排除

本目录包含 204 个服务器来源文件，来源映射、大小和 SHA-256 见 `manifest.json`。采集时逐文件校验传输哈希。

仅纳入源码、配置、依赖清单、说明、实验协议及指标汇总；不包含真实 `.env`/令牌、模型权重、checkpoint、图纸数据、训练样本、生成的 CAD 样本、虚拟环境、数据库、进程状态及原始运行日志。符号链接未跟随，评估目录从实际位置单独采集。

数据集原件仍保留在服务器。此上传不授予第三方数据或依赖的再分发许可。
