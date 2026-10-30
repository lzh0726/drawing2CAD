"""Consolidate audit evidence without modifying source datasets or training configs."""
from pathlib import Path
import json
import sys

root = Path(sys.argv[1]).resolve()
out = root / 'experiments/dataset-audit-20261024'
p = out / 'audit.json'
r = json.loads(p.read_text())
r['test'].update(json.loads((out/'fusion-corrected.json').read_text()))
unit_rows = json.loads((out/'step-unit-definitions.json').read_text())
for key, folder in [('deepcad','test100_gt_steps_deepcad'),('f360rec','test100_gt_steps_f360'),('f360rec_mdim','test100_gt_steps_f360'),('zerotocad1m','test100_gt_steps_zerotocad1m')]:
    rows = [x for x in unit_rows if '/'+folder+'/' in x['file']]
    r['test'][key]['unit_definition_combinations'] = sorted({','.join(sorted(set(x['length_units']))) for x in rows})
    r['test'][key].pop('unit_definitions', None)
    r['test'][key].pop('step_length_unit_counts', None)
r['split_overlap'] = json.loads((out/'split-overlap.json').read_text())
r['step_geometry_samples'] = json.loads((out/'step-geometry.json').read_text())
r['license_status'] = 'Unresolved: no license found in either ZIP; GitHub repo license=null. Do not infer dataset rights from dependency software licenses.'
r['scope_limits'] = ['No dataset Python executed','No GPU training or model inference','No claim of complete dimension/feature verification','No near-duplicate geometry detection','Static result=solid adaptation checked only; original labels untouched']
p.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
# Replace preliminary directory-mapping results with verified final evidence.
(out/'test-inspection.json').write_text(json.dumps({k:v for k,v in r['test'].items() if k!='directories'},ensure_ascii=False,indent=2),encoding='utf-8')
t=r['train']; overlap=r['split_overlap']
rows='\n'.join(f"| {k} | {r['test'][k]['rows']} | {r['test'][k]['nonempty_code_labels']} | {len(r['test'][k]['missing_images'])} | {len(r['test'][k]['missing_steps'])} |" for k in ['deepcad','f360rec','f360rec_mdim','zerotocad1m'])
hashes='\n'.join(f"- `{name}`: `{info['sha256']}`；CRC 首个坏文件：`{info['crc_first_bad_member']}`。" for name,info in r['archives'].items())
report=f'''# Ortho2CAD 数据准入检查报告

检查日期：2026-10-24。服务器：用户已授权的 Linux 算力服务器。范围：已下载的两个 ZIP、现有 server/policy.py、官方说明，以及本地 CadQuery 2.8.0 导入标准 STEP 的复核。

结论：可以继续做数据清洗与格式适配，但不能直接将原始包作为“已审核、可正式训练”的数据。发现标签输出变量不兼容、训练/测试代码重复、部分尺寸标注算法误差及许可未明确。没有启动训练，没有执行任何数据集 Python 标签，也没有改动原始 ZIP。

## 1. 文件与配对

训练 JSON：`ortho_train_data/cadquery_ortho_train_data.json`。
训练记录 {t['rows']:,} 条；PNG {r['archives']['ortho_train_data.zip']['extensions']['.png']:,} 张；训练引用唯一图片 {t['unique_images']:,} 张；缺失图片 {t['missing_image_count']}；异常对话记录 {t['malformed_count']}。
图片路径为 `image` 字段相对 `ortho_train_data/` 的路径。标签位于 `conversations` 中 `from=gpt` 的 `value`。
训练 ZIP 没有 STEP；未被当前训练 JSON 引用的图片 {t['unreferenced_png_count']:,} 张，不应通过遍历全部 PNG 自行推断其属于训练集。

| 测试组 | 记录数 | 非空代码标签 | 缺失图片 | 缺失 STEP |
|---|---:|---:|---:|---:|
{rows}

Fusion 与 Fusion 多尺寸图对应相同的 100 个三维零件。400 张测试 PNG 只有 300 个不同 STEP；普通 JSONL 和 `_api.jsonl` 也不是独立测试样本，不能叠加计数。
训练数据的存在配对已经全量核查，但存在文件不证明图纸和建模代码几何语义完全一致。

## 2. 代码兼容性

全量原始代码检查：`{json.dumps(t['raw_policy'],ensure_ascii=False)}`。
仅在内存中追加 `result = solid` 后检查：`{json.dumps(t['with_result_equals_solid_policy'],ensure_ascii=False)}`。
缺失 solid 赋值记录数：{t['missing_solid_assignment']}。导入统计：`{json.dumps(t['imports'])}`。

原始代码以 solid 为最终对象，本项目要求 result。后续转换时保留原程序并追加 `result = solid`，同时统一提示词。不要全局替换 solid，因为还存在 solid0 等中间变量。
静态检查通过不证明轮廓闭合、布尔操作成功、单实体或正体积。服务器当前未发现 docker 命令，训练 Conda 环境没有 CadQuery，因此未在服务器进行标签动态执行；不得把 AST 检查写成“CAD 执行成功率”。

## 3. 训练/测试泄漏

训练与 DeepCAD 100 条测试数据零件 ID 重复数：{len(overlap['same_part_id'])}。
但是其中 **{len(overlap['same_exact_code'])} 条测试代码与训练集逐字一致**，对应 **{overlap['matching_train_rows']:,} 条训练记录、{overlap['matching_test_code_hashes']} 个不同代码哈希**。
训练集自身逐字重复代码的冗余记录数：{t['exact_code_duplicate_rows']:,}。
证据：`split-overlap.json`、`train-exclusion-candidates.json`。这些清单仅标记，尚未删除或改写训练数据。
下一步：保留官方测试集；从训练候选中排除与测试标签同哈希的全部记录；在剩余记录中按代码指纹分组后划分 train/val。还需检测语法不同但几何相同或近似的样本；本次未完成几何近重复检测。

## 4. 单位、坐标与视图

所有 300 个 STEP 已检查单位声明。DeepCAD、ZeroToCAD 的长度声明为 MILLI/METRE；Fusion 文件同时含 CENTI/METRE 和 METRE 定义，不能按文本坐标直接当毫米，必须由 STEP 导入器处理其上下文。
本地导入 12 个 STEP 实例（9 个不同零件；3 个 Fusion 在普通图与多尺寸图中重复），均为有效单实体。见 `step-geometry.json`。这是标准答案的几何检查，不是标签代码执行检查。

例：deepcad-000 的包围盒约 0.671875 × 0.247533 × 0.031250 mm，与图中 0.6719、0.2475、0.0313 一致。不要擅自将这些数值放大 1000 倍。
Fusion 首个样本 STEP 导入后为 50.8 × 43.9942 × 34.93 mm，与图纸标注吻合。
Fusion 的另外两个样本原点偏移很大，图中没有对应绝对坐标标注。现有不做平移对齐的 IoU 可能惩罚正确但放置不同的重建。应预先规定坐标约定，或分别报告原坐标 IoU 与仅平移对齐 IoU；不能用缩放对齐掩盖尺寸错误。
现有统一 0.1mm 包围盒阈值对 0.03125mm 厚的 DeepCAD 样本过宽。应同时报告绝对与相对尺寸误差，并按数据尺度制定阈值。

查看的样图为第一角布局：右视图位于左，主视图位于右上，俯视图位于主视图下；标有 Right/Front/Top View。标准合成图多为外包围盒尺寸，不能等同完整制造图。训练图采样编号见 audit.json；视觉抽查不是逐图人工审核。

## 5. 发现并复现的尺寸标注误差

`zerotocad1m-001` 图中标注 X=78.0483、Y=82.4677、Z=80.0000；对同一个标准 STEP 的较紧包围盒计算得到约 X=59.8157、Y=59.6314、Z=80.0000。
使用官方绘图脚本中的 `BRepBndLib.Add` 算法复算，得到 78.048322、82.467738、80.000003，与图中文字相符；使用 `AddOptimal` 得到上述较紧范围。
`zerotocad1m-002` 也存在 X=80.1369（图中/宽包围盒）与约 80.0000（较紧范围）的差异。
证据：`bbox-method-comparison.json` 及对应 inspection 样图/STEP。这说明某些曲面零件的标注源于保守包围盒，而非真实紧包围盒尺寸；不应把模型与这些标注的冲突全部算作模型错误。后续需筛查受影响样本，或从 STEP 重新生成经过验证的标注。其他样本尚不能据此推定均有问题。

## 6. 许可证与用途记录

两个压缩包均没有 README/LICENSE/COPYRIGHT 文件。此次公开 GitHub API 检查：Ortho2CAD 与 GenCAD-Code 的 license 字段为 null，Ortho2CAD 仓库根目录无许可证文件。官方数据卡描述研究用途与论文引用，但本次未找到明确的数据许可条款；服务器访问 Hugging Face API 失败，不能把失败解释为授权不存在。
状态应填写 `license_status=unresolved`，不能填写自造的 MIT、Apache 或“已授权商用”。DeepCAD 代码仓库的 MIT 软件许可证不能自动证明派生 CAD 数据、图纸、标签的全部使用权。
本次只能确认公开下载入口与研究背景，无法确认训练、再分发、商用各自授权范围。正式使用或发布前应查明上游数据条件并取得明确许可。未发送任何联系作者的消息。

## 7. 下一阶段清单

1. 明确数据许可并保留出处；不要把许可未知样本标为已授权。
2. 按 test 代码哈希排除 3,240 条训练候选，再按指纹分组划分验证集；进行几何近重复检查。
3. 将 solid 输出适配为 result，图片路径转换为服务器路径，保持原始数值与坐标。
4. 建立隔离 CAD 执行环境，验证小批标签的实际执行、实体有效性和图纸一致性。
5. 检查曲面标注误差，并统一单位、坐标评估策略及误差阈值。
6. 完成上述准入检查后，再导出 SFT 数据和启动 20 步试训。本轮没有标记 human_reviewed=true。

## 8. 复现与证据

- 全量审计脚本：`tools/audit_ortho_dataset.py`（标准库，导入本项目 policy，不执行标签）。
- 标准 STEP 复核脚本：`tools/check_audit_step_geometry.py`（在已安装 CadQuery 的环境使用）。
- 结构化报告：`experiments/dataset-audit-20261024/audit.json`。
- 本报告：`docs/DATASET_AUDIT_20261024.md`。
- 包 SHA256 记录用于固定本次输入；并未声称与发布方签名校验。
{hashes}

官方来源：
- https://github.com/AdityaJoglekar/Ortho2CAD
- https://huggingface.co/datasets/AdityaJoglekar/Ortho2CAD_Orthographic_Drawings
- https://github.com/AdityaJoglekar/Ortho2CAD/blob/main/orthographic_drawing_generation/pythonocc_for_step_to_ortho.py
- https://github.com/anniedoris/GenCAD-Code
- https://github.com/rundiwu/DeepCAD/blob/master/LICENSE
'''
(root/'docs').mkdir(exist_ok=True)
(root/'docs/DATASET_AUDIT_20261024.md').write_text(report,encoding='utf-8')
print(json.dumps({'train':t,'overlap':overlap,'report':str(root/'docs/DATASET_AUDIT_20261024.md')},ensure_ascii=False,indent=2))
