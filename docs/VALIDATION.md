# 本地交付验收记录

验收日期：2026-10-24（Asia/Shanghai）。项目目录：`C:\Users\卢卓宏\Desktop\drawing2CAD`。

## 实际执行结果

| 检查 | 结果 |
|---|---|
| 后端集成测试 | `8 passed in 15.11s`，使用真实 CadQuery/OpenCascade |
| Python 依赖检查 | `pip check` → `No broken requirements found.` |
| 前端 | TypeScript 检查和 Vite production build 成功 |
| Windows PowerShell | `powershell.exe` 运行前端构建、启动、停止均成功 |
| HTTP 页面 | `GET /` 返回 200；`/api/health` 正常 |
| 重复启动 | 检测到已有的本项目服务，未启动第二个实例 |
| 重启持久化 | 实际停止后再次启动，任务 completed，仍保留 2 个版本 |
| 浏览器真实 CAD | 安装板实体可见；俯视图显示中心孔与 4 个安装孔 |
| 浏览器对话修改 | 输入“厚度改为 15，孔径改为 24”，产生 v2，尺寸显示 80 × 60 × 15 mm |
| 浏览器版本切换 | v1 仍显示 80 × 60 × 12 mm，源码孔径 20；最新版本为 v2 |
| 浏览器连接设置 | 未配置服务器时明确显示未连接及对接步骤 |
| 浏览器控制台 | 初次正常生成和对话修改后未发现 error 日志；主动停止服务测试会产生预期网络断开 |

## 后端测试覆盖

1. 上传图纸、预览、SQLite 持久化；无服务器时不产生虚假模型。
2. 无效文件、空文件、无效图片页码和非本地 Origin/Host 拒绝。
3. 两页 PDF 选择第二页并检查输出比例，拒绝超范围页码。
4. 真实 STEP/STL/源码/报告输出；下载 STEP 后用 CAD 内核重新导入；修改尺寸和孔径；v1 文件逐字节不变；非法参数与未知版本拒绝。
5. 运行中的重复请求拒绝；取消任务后无成功版本。
6. 服务重启时将未完成任务标记为 interrupted。
7. ZIP 响应仅读取白名单文件，忽略路径穿越条目，拒绝无效 ZIP；不执行返回源码。
8. 通过 HTTPX MockTransport 验证远端 multipart 请求与 ZIP 响应，再真实导入 STEP、生成预览。

第 8 项使用构造的几何实体，不是 GPU 模型推理验证。

## 已验证环境

- Python 3.12.14
- FastAPI 0.161.1、Uvicorn 0.54.0
- CadQuery 2.8.0、LangGraph 1.2.16、pytest 9.1.1
- 前端 Vite 7.3.7；最终构建由本机 bundled Node.js 24.22.1 执行
- 完整版本记录：`backend/requirements.lock.txt`、`frontend/package-lock.json`

Vite 报告主 JS 包约 759 KB（gzip 约 208 KB）的体积提示，构建成功；当前面向本地使用，尚未进行首屏性能优化。

## 未完成于本地服务范围内的事项

未部署、训练或验收真实二维工程图重建模型；未验证外部 GPU 服务器、真实模型准确率、多视图/尺寸一致性、装配体与多用户公网部署。当前提供模型服务调用适配层和说明，不将几何有效性当作还原准确率。

## 页面截图

![工作台：本地生成的 v2 安装板](workbench.png)
