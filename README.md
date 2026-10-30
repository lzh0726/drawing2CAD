# Drawing2CAD 本地工作台

> 仓库说明（2026-10-30）：根目录包含本地前后端源码；服务器上的代码、A2/A3 训练评估及 A4 数据准备脚本另存于 [server-snapshot/](server-snapshot/README.md)，保留双方版本。下文和 `docs/` 中的阶段记录保留编写时的状态，不代表服务器当前实验进度。数据集、模型权重、任务数据库、运行日志及真实连接凭据不随源码发布；部分训练工具包含原开发环境路径，使用前需按部署环境调整。首次使用请按第 3 节安装依赖并构建前端。

本项目实现本地前端、任务后端、Agent 工作流、CAD 预览和服务器推理接口。技术栈为 React + TypeScript + Three.js、FastAPI + LangGraph、SQLite、CadQuery/OpenCascade。

**当前边界：本地参数化样例可以真实生成三维实体；任意二维图纸自动重建需要另行部署 GPU 建模服务器。这里没有附带或训练 AI 权重，也不会用样例冒充图纸识别结果。**

## 1. 立即使用

1. 双击根目录 `启动服务.cmd`，浏览器打开 <http://127.0.0.1:8000>。
2. 点击“打开样例”，等待本机 CadQuery 生成安装板。
3. 拖动旋转、滚轮缩放；切换俯视、正视、线框、图纸和源码。
4. 在对话中输入“厚度改为 15，孔径改为 24”，或者修改参数面板后运行。完成后保存为新版本，旧版本可切换和下载。
5. 下载 STEP、STL、源码；查看几何检查结果。
6. 用 `停止服务.cmd` 结束后台服务。关闭网页不会停止服务。修改 `.env` 后需要停止再启动。

正常使用只启动一个 8000 端口：FastAPI 同时提供编译好的前端和后端 API。无需一直运行 Node，也不需要本地 GPU。默认只监听本机，不向局域网开放。

## 2. 已实现的功能

| 模块 | 行为 |
|---|---|
| 图纸上传 | PNG/JPG/PDF；最大 20 MB；PDF 最多 100 页，上传前选择页码；图片最多 3000 万像素 |
| 数据保存 | 原始文件、选中页预览、任务、对话、事件和模型版本保存在 `storage/` |
| 工作流 | LangGraph 依次准备输入 → 生成/调用远端 → 几何检查 → 发布版本 |
| 执行管理 | 单任务串行队列、实时 SSE 事件、取消、失败信息、重启后中断状态恢复 |
| 3D 显示 | Three.js 显示由实际 STEP 实体产生的 STL，支持旋转、缩放、视角和线框 |
| 参数样例 | 5 孔安装板，支持长、宽、厚度、中心孔径；本地确定性模板和文字参数识别 |
| 几何检查 | 单实体、OpenCascade 有效性、正体积、包围盒；样例额外检查外形尺寸 |
| 版本 | 成功结果追加新版本；失败不覆盖旧版本；按版本下载 STEP/STL/Python/JSON |
| 服务器适配 | 可配置地址和令牌，上传图纸及对话，接收 STEP 和源码，重新生成 STL |

“几何有效”不能证明“与图纸一致”。系统明确将图纸符合性标记为待验证；还没有 OCR 尺寸约束、自动多视图一致性判定、装配体重建或自主修复循环。当前 Agent 是可观察的工具工作流；本地样例对话不是大模型推理。

上传真实图纸但未配置服务器时，任务显示“等待服务器”，文件仍会保存。配置服务器后点击“开始建模”。目前输入以选定页面整体 PNG 传送，未自动拆分三视图。

## 3. 环境安装与开发

此目录已配置 Python 虚拟环境和编译产物。复制到其他电脑时重新安装，不要直接搬运 `.venv`。

建议环境：Windows、Python 3.12、Node.js 22.12+ 或 24、npm。项目脚本会优先使用本机 Codex 附带的运行时，再回退到系统运行时；不会替换系统 Python/Node。

在项目根目录 PowerShell 执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

安装使用 `backend/requirements.lock.txt` 和 `frontend/package-lock.json` 锁定已验证版本。首次安装 CAD 依赖下载较大，请耐心等待。

前后端热更新开发，分别打开两个终端：

```powershell
# 终端 1；先停止由启动脚本启动的服务，释放 8000
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# 终端 2；Vite 5173 将 /api 代理到 8000
powershell -ExecutionPolicy Bypass -File .\scripts\frontend.ps1 dev
```

开发页面 <http://127.0.0.1:5173>；API 文档 <http://127.0.0.1:8000/docs>。前端改完后执行 `scripts/frontend.ps1 build`，重新启动正式服务使用新产物。不要启用多个 Uvicorn worker：当前队列属于单进程。

## 4. 连接 GPU 服务器

详细协议见 [docs/REMOTE_API.md](docs/REMOTE_API.md)。需要一个能将模型推理结果转换为实际 CAD 产物的适配器；不能直接把地址指向普通 vLLM/OpenAI chat-completions 服务。

1. 在 GPU 服务器部署符合协议的适配器，监听 `127.0.0.1:8001`。
2. 本地终端建立隧道（将 USER、SERVER 替换为自己的配置）：

```powershell
ssh -N -L 9000:127.0.0.1:8001 USER@SERVER
```

3. 复制 `.env.example` 为 `.env`，编辑：

```dotenv
CAD_REMOTE_URL=http://127.0.0.1:9000
CAD_REMOTE_TOKEN=
CAD_REMOTE_TIMEOUT=600
```

认证令牌只填入本机 `.env`，不提交、不粘贴进聊天。无需认证时留空。修改后重启后端，打开页面右上角连接设置检查状态。只有点击真实图纸的运行/对话修改，才会把该任务预览图及对话发给你配置的远端。

## 5. 目录与接口

```text
backend/app/main.py        HTTP API、上传、SSE、静态前端
backend/app/worker.py      LangGraph、队列、远端调用、版本发布
backend/app/cad_runner.py  可信模板和 STEP 导入检查的独立进程
backend/app/store.py       SQLite 元数据和事件
backend/tests/test_api.py  集成测试
frontend/src/App.tsx       工作台和 API 交互
frontend/src/ModelViewer.tsx Three.js 预览
scripts/                  安装、构建、启动和停止
storage/jobs/<id>/input/   原始图纸和预览
storage/jobs/<id>/attempts/<id>/  每次执行的 CAD 产物
storage/jobs.sqlite3       任务、对话和事件
storage/logs/              本地服务日志
```

核心接口：`GET /api/health`、`GET /api/connection`、`GET/POST /api/jobs`、`POST /api/demo`、`GET /api/jobs/{id}`、`POST /api/jobs/{id}/run`、`POST /api/jobs/{id}/messages`、`POST /api/jobs/{id}/cancel`、`GET /api/jobs/{id}/events`、`GET /api/jobs/{id}/artifacts/{step|stl|code|report}?version=1`。

## 6. 验证与限制

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
.\.venv\Scripts\python.exe -m pip check
powershell -ExecutionPolicy Bypass -File .\scripts\frontend.ps1 build
```

测试会在临时目录真实运行 CadQuery，包括下载 STEP 后重新导入、尺寸修改、历史版本不变、非法上传、PDF 选页、取消、重启恢复和模拟远端协议。模拟 HTTP 适配器不等于真实 GPU 推理验证。实际验收记录见 [docs/VALIDATION.md](docs/VALIDATION.md)。

本地不执行远端返回的 Python 源码；仅将其存档和展示。STEP 导入仍使用本机原生 CAD 库，独立进程和超时不构成完整安全沙箱。此版本用于单用户本地开发，不适合直接公开部署。取消会停止本地等待/本地 CAD 子进程；远端服务器是否停止计算取决于远端实现，当前协议未提供远端取消接口。

任务数据无自动清理。备份时先停止服务，再复制 `storage/`；无需备份 `.venv`、`node_modules`。若端口已被占用，启动脚本会提示并退出，不会结束不属于本项目的进程。

## 服务器端增量（2026-10-24）

新增 server/：Qwen3-VL 候选推理适配器、Docker CAD 执行、最多两次修复、审计记录、固定测试集评估和人工审核训练数据导出。服务器安装与接入步骤见 [server/README.md](server/README.md)。代码已写入本目录；实际 GPU 部署和真实图纸验收仍待 SSH 密钥认证完成。

32 个合成回归样例位于 datasets/smoke32/，全部为 test，不能代替真实工程图测试集。源码部署包为 drawing2cad-server-source.zip。
