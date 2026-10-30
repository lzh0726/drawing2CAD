"""Render completed metrics into a compact report; no model or CAD execution."""
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


def main(folder):
    summary_file=next(p for p in [folder/'summary-serial.json',folder/'summary-final.json',folder/'summary-resolved.json',folder/'summary.json'] if p.exists())
    summary=json.loads(summary_file.read_text())
    protocol=json.loads((folder/'protocol.json').read_text())
    assert summary['status']=='completed' and summary['completed_count']==44
    baseline=summary['baseline']
    rows=[]
    for ident in protocol['evaluated_ids']:
        a=json.loads((folder/'samples'/(ident+'.json')).read_text())
        b=json.loads((folder/'baseline-first-pass'/(ident+'.json')).read_text())
        for group,record in [('a2',a),('baseline',b)]:
            for directory in ['metric-retries','metric-cache-equivalence','serial-metrics']:
                retry=folder/directory/(group+'-'+ident+'.json')
                if retry.exists():
                    resolved=json.loads(retry.read_text())
                    if 'metrics' in resolved:
                        record['metrics']=resolved['metrics'];record.pop('metric_error',None)
        m=a.get('metrics',{}); bm=b.get('metrics',{})
        rows.append({'id':ident,'a1_valid':b['valid_solid'],'a2_valid':a['valid_solid'],
            'a1_iou_raw':bm.get('iou_raw'),'a2_iou_raw':m.get('iou_raw'),
            'a1_iou_translation_aligned':bm.get('iou_translation_aligned'),
            'a2_iou_translation_aligned':m.get('iou_translation_aligned'),
            'a2_bbox_relative_xyz':json.dumps(m.get('bbox_relative')),
            'a2_seconds':a['seconds'],
            'a2_failure':a.get('attempts',[{}])[0].get('error',a.get('attempts',[{}])[0].get('status')),
            'metric_error':a.get('metric_error','')})
    with (folder/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def value(obj,key):
        v=obj.get(key)
        if v is None:return '尚无完整指标'
        if key.endswith('rate'):return f'{v:.2%}'
        return f'{v:.6f}' if isinstance(v,float) else str(v)
    metrics=[('有效单实体数量','valid_solid_count'),('有效单实体率','valid_solid_rate'),
             ('原坐标 IoU，失败记 0','mean_iou_raw_all'),
             ('仅包围盒中心平移 IoU，失败记 0','mean_iou_translation_aligned_all'),
             ('成功且可比较实体的平移 IoU','mean_iou_translation_aligned_successful_comparable'),
             ('固定 XYZ 外包尺寸相对误差 ≤1% 数量','bbox_xyz_within_1pct_count'),
             ('固定 XYZ 外包尺寸相对误差 ≤5% 数量','bbox_xyz_within_5pct_count')]
    finished=datetime.fromisoformat(summary.get('metrics_finished_at',summary['finished_at'])).astimezone(ZoneInfo('Asia/Shanghai')).isoformat()
    lines=['# A2 固定测试集评估','',f'完成时间（北京时间）：{finished}','',
        '| 指标 | A1：1,000 条训练 | A2：10,000 条训练 |','|---|---:|---:|']
    lines += [f'| {label} | {value(baseline,key)} | {value(summary,key)} |' for label,key in metrics]
    lines += ['', '## 协议与边界','',
        '- 固定 50 个测试零件，沿用 6 个无效参考实体排除项，指标分母为 44。',
        '- 两模型均为冻结视觉编码器和投影层、语言骨干 LoRA rank16；A2 使用第 1200 步最佳适配器。',
        '- 首次生成评估，无自动修复；16384 新 token 上限、600 秒生成时限、262144 图像像素上限、贪心解码。',
        '- A1 使用 2026-10-27 当前预算评估的首轮结果和 STEP 缓存，本次重新计算几何指标；未使用其后续修复输出。',
        '- 生成的 Python 仅在 Bubblewrap 内执行；CPU 35 秒、CAD 墙钟 90 秒；独立输出目录。',
        '- STEP 指标初次比较为 CPU 35 秒/墙钟 90 秒；失败后独立复核为 CPU 120 秒/墙钟 300 秒。原始失败记录保留。',
        f'- 最终计算方式：{summary.get("metric_execution_mode","初次计算及独立复核")}。串行复核预算为 CPU 120 秒/墙钟 180 秒；IoU 公式、尺度、对齐策略均不变。',
        '- IoU 采用精确布尔交并，报告原坐标和包围盒中心平移两种口径，不旋转、不缩放。不是论文归一化/姿态对齐口径。',
        '- 参考 STEP、代码标签、人工修复代码未送入模型。A2 训练数据与测试数据编号及家族分组无交集。',
        '- 本测试集已用于历史故障诊断，定位为回归集；不能替代新的独立泛化测试。',
        '- 尺寸 1%/5% 是诊断阈值，不能当作制造公差；尺寸沿用数据源归一化尺度。',
        f'- 原始来源未变化：{summary["sources_unchanged"]}。',
        f'- A2 IoU 计算错误：{summary["metric_error_ids"]}；A1：{baseline["metric_error_ids"]}。',
        (f'- A1 历史指标复用样本：{summary["baseline_historical_metric_reuse_ids"]}；核验文件见 metric-cache-equivalence/。'
         if summary.get('baseline_historical_metric_reuse_ids') else '- 最终指标全部来自本次串行重算，没有使用历史指标填补缺失值。'),
        '', '## 样本变化','',
        f'- 新增有效实体：{summary["newly_valid_ids"]}',
        f'- 原成功、现失败：{summary["regressed_ids"]}',
        '', '## 每个样本','',
        '| 样本 | A1 有效 | A2 有效 | A1 平移 IoU | A2 平移 IoU | A2 耗时/秒 |',
        '|---|---|---|---:|---:|---:|']
    for r in rows:
        def metric_text(key,valid):
            v=r[key]
            return f'{v:.6f}' if v is not None else ('计算失败' if valid else '0（无效）')
        lines.append(f'| {r["id"]} | {r["a1_valid"]} | {r["a2_valid"]} | {metric_text("a1_iou_translation_aligned",r["a1_valid"])} | {metric_text("a2_iou_translation_aligned",r["a2_valid"])} | {r["a2_seconds"]:.1f} |')
    lines += ['', '## 失败记录', '', '| 样本 | 执行器记录 |', '|---|---|']
    for r in rows:
        if not r['a2_valid']:
            error=r['a2_failure'].replace('\n',' ').replace('|','/')
            lines.append(f'| {r["id"]} | {error} |')
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(str(folder/'REPORT.md'))


if __name__=='__main__':
    main(Path(sys.argv[1]))
