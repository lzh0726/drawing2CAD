# A6000 服务器接入与验收

本目录实现 Qwen3-VL 推理、CadQuery 代码检查、Docker 执行、最多两次修复、结果打包，以及固定测试集评估。当前基座是候选 **Qwen/Qwen3-VL-8B-Instruct**，未进行 CAD 专项微调；不要把基座输出视为工业精度结果。

## 当前状态与验证边界

本机可测试 HTTP 协议、修复分支、参数编辑上下文、错误脱敏、真实 STEP 导入和评估程序。模型权重尚未下载到目标服务器，GPU 推理、Docker 镜像构建和真实图纸成功率尚未实测。没有 Docker 时服务会拒绝启动，不回退到宿主机执行模型代码。

模型加载接口参考工作区已保存的 Ortho2CAD 官方代码 `src/qwenvl/eval/model_vqa_loader.py`；当前联网资料检查不可用，因此没有宣称候选依赖组合已经在 A6000 验证。部署后必须保存真实环境锁和模型 revision。

## 1. 登录与准备目录

目标连接由用户提供：

```powershell
ssh -p 10003 luzhuohong@14.153.254.201
```

在交互式终端输入密码，或使用已配置的 SSH 密钥。不要把密码放进脚本、命令参数、聊天或环境示例。仅使用已分配给你的 GPU；有调度系统时先申请计算资源。

在服务器用户目录准备 `~/drawing2CAD`，复制本地的 `server/`、`backend/`。不需要上传 Windows 的 .venv、node_modules、.env、运行记录或模型缓存。

本机传输命令（需要已完成 SSH 认证）：

```powershell
ssh -p 10003 luzhuohong@14.153.254.201 "mkdir -p ~/drawing2CAD"
scp -P 10003 -r server backend .dockerignore luzhuohong@14.153.254.201:~/drawing2CAD/
```

也可使用本次生成的服务器源码压缩包；仅包含源码、文档和测试，不含凭据和环境。

## 2. 预检

```bash
cd ~/drawing2CAD
bash server/preflight.sh
```

确认 Python 3.12 可用、GPU 空闲显存、驱动、磁盘以及你有权使用的 Docker。脚本不会安装 Docker、修改驱动或申请管理员权限。若没有可用容器运行时，先请管理员配置；不要关闭隔离来凑通流程。

8B BF16 权重约 16 GB 量级，仍需视觉特征、KV cache 和运行时显存。单张 48 GB 是候选推理配置，不是已测峰值；先跑一张图再提高上下文长度。准备足够的模型、依赖和 Docker 镜像空间，建议预留约 50 GB 后再安装。

## 3. 用户目录内安装

```bash
bash server/setup.sh
```

安装到独立的 `.venv-server`，候选 Torch 2.8/CUDA 12.6、Transformers 4.57 系列；实际驱动不兼容时选择对应 Torch wheel，不能修改共享服务器驱动。通过 `TORCH_INDEX_URL` 可指定已确认的官方 wheel 地址。脚本先检查 CUDA，再构建 CPU CAD 镜像。

该脚本生成随机 API 令牌到 `server/server.env`，权限 600，不在控制台打印。已有 server.env 不覆盖。实际依赖写入 `server/requirements.linux.lock.txt`。完整 CUDA/GPU 可用性还需要运行实测确认。

## 4. 单卡启动

由调度系统设置 CUDA_VISIBLE_DEVICES；无调度系统时，仅在获得对应 GPU 使用权后设置该变量。

```bash
bash server/start.sh
```

首次启动下载模型。默认监听服务器 `127.0.0.1:8001`，仅一个 worker；首次加载完成后才提供请求处理。需要持续运行时，在你已有的 tmux 会话内启动，Ctrl+C 停止。

输入图最长边限制为 1600 像素，输出最多 2048 tokens。模型生成设置 180 秒软时限，CAD 容器有独立 90 秒硬时限，最多初次生成加两次修复。软时限不是强制中断 GPU 内核的保证。复杂零件源码被截断时，应先检查失败记录，再明确增加 token 预算并实测显存。

## 5. 接回本地页面

本机终端保持以下 SSH 隧道：

```powershell
ssh -N -p 10003 -L 9000:127.0.0.1:8001 luzhuohong@14.153.254.201
```

在本地项目 .env 中填写：

```dotenv
CAD_REMOTE_URL=http://127.0.0.1:9000
CAD_REMOTE_TOKEN=YOUR_SERVER_TOKEN_HERE
CAD_REMOTE_TIMEOUT=900
```

令牌与服务器 server.env 一致，使用本地编辑器安全配置，不经聊天转发。重启本地服务，打开右上角连接设置，再上传一张有明确尺寸的简单工程图。先人工核对长、宽、厚、孔径、孔数量和位置。

## 6. 修复与审计

每个远端任务有 UUID，保存于 server-storage/<id>/：

- drawing.png、request.json：本次输入。
- attempt-N.txt、attempt-N.py：模型原始输出与提取代码。
- trace.json：每次成功/失败、耗时、模型 revision、峰值显存。
- model.step、model.py：最终成功结果。

返回 ZIP 包含 model.step、model.py 和 generation.json。旧版本地接口可忽略 generation.json，仍能正常预览和下载。后续修改携带上次成功代码和用户对话。失败候选不会覆盖上次成功版本。

模型明确返回 NEEDS_CLARIFICATION 时不会虚构实体；服务返回 422 和 run_id，问题记录于 trace.json。当前本地 UI 对远端错误只显示状态码，详细澄清问题需查看服务端记录后通过对话补充；尚未实现自动澄清弹窗。

## 7. 固定数据集与评估

先生成固定的 32 个模板样例，用于系统回归：

```powershell
.\.venv\Scripts\python.exe -m server.make_dataset --output datasets/smoke32 --count 32
```

每项包含图纸、STEP、源码、参数、来源和 SHA256。**这些是本项目生成的安装板示意图，全部来自同一个模板家族，全部标记为 test；它们不能替代真实工程图基准，也不能拆分到训练集和测试集两侧。**

接入服务器后运行：

```bash
python -m server.evaluate --manifest datasets/smoke32/manifest.jsonl --output experiments/baseline-001 --url http://127.0.0.1:8001 --limit 1
# 第一张人工验收通过后，去掉 --limit 1 扩大测试。
```

评估依赖本地 CadQuery，建议在已有本地 .venv 下通过 SSH 隧道评估；Linux 推理环境默认不装宿主机 CAD，若在 Linux 执行评估应另建 CPU 评估环境。令牌由环境变量 CAD_SERVER_TOKEN 或 CAD_REMOTE_TOKEN 提供，不用命令参数传入。

结果记录有效实体率、XYZ 包围盒误差、同坐标系实体 IoU、修复次数、耗时及模型 revision。IoU 不做缩放、平移或旋转对齐；坐标不一致会降低结果，需要在基准中统一坐标约定。包围盒误差不是孔径等全部关键尺寸误差，孔槽特征仍需人工核验。不可比较的 IoU 记为空，并单独报告分母。

## 8. 获取真实评测数据，之后再训练

已有 Ortho2CAD 官方仓库 README 提供数据入口：
https://huggingface.co/datasets/AdityaJoglekar/Ortho2CAD_Orthographic_Drawings/tree/main

先获取 inference.zip，检查数据许可和图纸/STEP 配对，再选定 30～50 个固定测试零件。此轮未下载该文件，也未验证当前链接可访问性。不要把这些测试样本拿去微调。私人图纸需有使用授权。

真实 manifest 每行最少包含：
```json
{"id":"part-001","drawing":"images/part-001.png","step":"steps/part-001.step","group":"original-part-family-001","split":"test","source":"dataset name and original id","license":"license reference","instructions":"Reconstruct the dimensioned part."}
```

先分析实际失败，再收集训练集。对于训练标签，补充 code 路径，人工检查一致性后标记 human_reviewed=true，并设置 split=train。导出脚本会拒绝同 group 跨训练/测试泄漏：

```bash
python -m server.export_sft --manifest datasets/reviewed/manifest.jsonl --output datasets/sft/train.jsonl
```

这是通用图文对话 JSONL，不是某训练框架的即插即用配置。尚未训练或宣称微调改善；等有基线结果后再选择 LoRA 参数和训练框架。

## 9. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests server/tests -q
```

服务器已有 Docker 镜像后，增加真实容器测试：

```bash
CAD_DOCKER_TESTS=1 .venv-server/bin/python -m pytest server/tests -q
```

完整 server/tests 的 STEP fixture 需要 CadQuery；GPU .venv-server 默认不安装它。可在单独 CPU 测试环境安装 backend/requirements.txt 后执行完整测试，或先直接执行 Docker 的单项测试：

```bash
CAD_DOCKER_TESTS=1 .venv-server/bin/python -m pytest server/tests/test_server.py::test_actual_container -q
```

Docker 限制包括无网络、无主机目录挂载、只读根目录、非 root、删除 capabilities、CPU/内存/PID/文件大小与时间限制。AST 检查只用于限定建模语言，不构成安全边界；Docker 也不等于对所有内核漏洞的绝对防护。

修复对照实验：将 server/server.env 中 CAD_MAX_REPAIRS 分别设为 0 和 2，重启服务，对同一份固定 manifest 分别运行评估并保存到不同实验目录。不能用同一轮修复后的成功率代替无修复基线。
