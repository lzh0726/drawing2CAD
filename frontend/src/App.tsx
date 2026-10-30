import { useEffect, useRef, useState } from 'react'
import { ArrowDownToLine, ArrowRight, ArrowUp, Box, Check, CheckCircle2, ChevronDown, Circle, CircleAlert, Clock3, Code2, FileImage, FolderClock, Layers3, LoaderCircle, Maximize2, Network, Plus, Settings2, SlidersHorizontal, Square, Upload, X } from 'lucide-react'
import ModelViewer from './ModelViewer'
import { active, api, artifact, json, statusText } from './types'
import type { Connection, Job, JobSummary, Params } from './types'

const defaults:Params={length:80,width:60,thickness:12,hole:20}
const paramFields:{key:keyof Params;label:string;min:number;max:number}[]=[{key:'length',label:'长度',min:50,max:200},{key:'width',label:'宽度',min:40,max:160},{key:'thickness',label:'厚度',min:5,max:40},{key:'hole',label:'中心孔径',min:4,max:60}]

export default function App(){
  const [jobs,setJobs]=useState<JobSummary[]>([])
  const [job,setJob]=useState<Job|null>(null)
  const [jobId,setJobId]=useState<string|null>(null)
  const [versionNo,setVersionNo]=useState(0)
  const [params,setParams]=useState<Params>(defaults)
  const [tab,setTab]=useState<'model'|'drawing'|'code'>('model')
  const [notice,setNotice]=useState('')
  const [pending,setPending]=useState(false)
  const [serverConfigured,setServerConfigured]=useState(false)
  const [apiOnline,setApiOnline]=useState(false)
  const [modal,setModal]=useState<'connection'|'history'|null>(null)
  const [connection,setConnection]=useState<Connection|null>(null)
  const [draft,setDraft]=useState('')
  const [instructions,setInstructions]=useState('')
  const [page,setPage]=useState(1)
  const [drag,setDrag]=useState(false)
  const [code,setCode]=useState('')
  const [allEvents,setAllEvents]=useState(false)
  const fileInput=useRef<HTMLInputElement>(null)
  const parametersKey=useRef('')
  const selected=job?.versions.find(v=>v.number===versionNo)||job?.versions.at(-1)
  const running=!!job&&active(job.status)
  const locked=pending||running
  const list=()=>api<JobSummary[]>('/jobs').then(setJobs)
  const choose=(id:string)=>{setJobId(id);setVersionNo(0);setTab('model');setModal(null);setNotice('')}
  useEffect(()=>{
    api<{remote_configured:boolean}>('/health').then(h=>{setApiOnline(true);setServerConfigured(h.remote_configured)}).catch(()=>setNotice('本地后端未连接，请使用启动脚本运行服务。'))
    list().then(()=>{}).catch(()=>{})
    api<JobSummary[]>('/jobs').then(items=>{const saved=localStorage.getItem('drawing2cad-job');setJobId(items.find(j=>j.id===saved)?.id||items[0]?.id||null)}).catch(()=>{})
  },[])
  useEffect(()=>{
    if(!jobId){setJob(null);return}
    localStorage.setItem('drawing2cad-job',jobId)
    let disposed=false,timer:ReturnType<typeof setTimeout>|undefined
    const update=()=>api<Job>(`/jobs/${jobId}`).then(j=>{if(!disposed){setJob(j);setApiOnline(true);setNotice(n=>n==='本地服务暂时断开，正在重连…'?'':n);const key=j.id+JSON.stringify(j.parameters);if(j.parameters&&parametersKey.current!==key){setParams(j.parameters);parametersKey.current=key}}}).catch(()=>{if(!disposed){setApiOnline(false);setNotice('本地服务暂时断开，正在重连…')}})
    update()
    const events=new EventSource(`/api/jobs/${jobId}/events`)
    events.onmessage=()=>{clearTimeout(timer);timer=setTimeout(()=>{update();list().catch(()=>{})},120)}
    const poll=setInterval(update,4000)
    return()=>{disposed=true;events.close();clearInterval(poll);clearTimeout(timer)}
  },[jobId])
  useEffect(()=>{
    if(tab!=='code'||!job||!selected)return
    const controller=new AbortController();setCode('正在加载源码…')
    fetch(artifact(job.id,'code',selected.number),{signal:controller.signal}).then(r=>{if(!r.ok)throw new Error('源码加载失败');return r.text()}).then(setCode).catch(e=>{if(e.name!=='AbortError')setCode(e.message)})
    return()=>controller.abort()
  },[tab,job?.id,selected?.number])
  const act=async(fn:()=>Promise<unknown>)=>{setPending(true);setNotice('');try{await fn();await list()}catch(e){setNotice(e instanceof Error?e.message:'操作失败')}finally{setPending(false)}}
  const openDemo=()=>act(async()=>{const j=await api<Job>('/demo',json(defaults));setJob(j);choose(j.id)})
  const upload=(file?:File)=>{if(!file)return;act(async()=>{const form=new FormData();form.append('file',file);form.append('instructions',instructions);form.append('page',String(page));const j=await api<Job>('/jobs',{method:'POST',body:form});setJob(j);choose(j.id)})}
  const run=()=>{if(!job)return;act(async()=>{setVersionNo(0);const j=await api<Job>(`/jobs/${job.id}/run`,json(job.kind==='demo'?{parameters:params}:{}));setJob(j);setTab('model')})}
  const send=()=>{if(!job||!draft.trim())return;const text=draft;act(async()=>{setVersionNo(0);await api<Job>(`/jobs/${job.id}/messages`,json({content:text}));setDraft('');setJob(await api<Job>(`/jobs/${job.id}`))})}
  const testConnection=async()=>{setConnection(null);try{const c=await api<Connection>('/connection');setConnection(c);setServerConfigured(c.configured)}catch{setConnection({configured:false,reachable:false,message:'本地后端未连接。'})}}
  const showConnection=()=>{setModal('connection');testConnection()}
  const newTask=()=>{setJobId(null);setJob(null);setVersionNo(0);setInstructions('');setNotice('');localStorage.removeItem('drawing2cad-job')}
  const drawingUrl=job?`/api/jobs/${job.id}/drawing${selected?`?version=${selected.number}`:''}`:null
  const displayEvents=job?.events||[]

  return <div className="app-shell">
    <aside className="rail"><button className="brand-mark" title="Drawing2CAD" onClick={newTask}><Layers3 size={24}/></button><button className="rail-button selected" title="建模工作台" onClick={()=>setModal(null)}><Box size={21}/></button><button className="rail-button" title="任务历史" onClick={()=>{list();setModal('history')}}><FolderClock size={21}/></button><button className="rail-button" title="服务器连接" onClick={showConnection}><Network size={21}/></button><div className="rail-bottom"><span className="rail-divider"/><button className="rail-button" title="服务设置" onClick={showConnection}><Settings2 size={20}/></button><div className="avatar">L</div></div></aside>
    <div className="app-main"><header className="topbar"><div className="wordmark">Drawing<span>2</span>CAD <span className="product-tag">WORKSPACE</span></div><div className="top-actions"><span className="local-status"><i className={apiOnline?'dot green':'dot'}/>{apiOnline?'本地服务运行中':'连接本地服务'}</span><button className="server-button" onClick={showConnection}><Network size={14}/>{serverConfigured?'服务器已配置':'连接建模服务器'}<ArrowRight size={13}/></button></div></header>
    <main className="page"><div className="page-heading"><div><div className="breadcrumb">工作空间 <span>/</span> 建模工作台</div><h1>从图纸，到模型<span className="title-period">.</span></h1><p>用自然语言构建与迭代，让每一步建模都清晰可见。</p></div><div className="heading-buttons"><button className="secondary" onClick={newTask}><Plus size={16}/> 新建任务</button><button className="primary" disabled={!job||!selected} onClick={()=>{if(job&&selected)window.location.href=artifact(job.id,'step',selected.number)}}><ArrowDownToLine size={16}/> 导出 STEP</button></div></div>
    {notice&&<div className="notice" role="alert"><CircleAlert size={17}/><span>{notice}</span><button onClick={()=>setNotice('')} aria-label="关闭提示"><X size={16}/></button></div>}
    <div className="workspace">
      <section className="panel input-panel"><div className="panel-heading"><span><FileImage size={17}/> 图纸输入</span><span className="step-label">01</span></div>
        <div className="upload-body"><input ref={fileInput} type="file" accept=".png,.jpg,.jpeg,.pdf" hidden onChange={e=>{upload(e.target.files?.[0]);e.target.value=''}}/>
        <button className={`dropzone ${drag?'dragging':''}`} disabled={locked} onClick={()=>fileInput.current?.click()} onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);if(!locked)upload(e.dataTransfer.files[0])}}>
          {job&&((job.kind==='demo'&&selected)||job.kind==='drawing')?<><img className="drawing-thumb" src={drawingUrl!} alt="当前工程图预览"/><span className="replace-drawing"><Upload size={14}/> 更换图纸</span></>:<><span className="upload-icon"><Upload size={23}/></span><strong>拖入二维工程图</strong><span>或点击选择文件</span><small>PNG、JPG、PDF · 最大 20 MB</small></>}
        </button>
        {job&&<div className="file-meta"><FileImage size={15}/><span title={job.title}>{job.title}</span><CheckCircle2 size={14}/></div>}
        <div className="form-row"><label htmlFor="page">PDF 页码</label><input id="page" type="number" min="1" max="100" value={page} onChange={e=>setPage(Number(e.target.value)||1)}/><span className="unit-label">单位 mm</span></div>
        <label className="field-label" htmlFor="requirements">补充要求 <span>可选 · 上传时保存</span></label><textarea id="requirements" className="instructions" placeholder="例如：单个零件，厚度 12 mm…" value={instructions} maxLength={2000} onChange={e=>setInstructions(e.target.value)}/>
        <div className="input-hint"><CircleAlert size={14}/><span>优先使用清晰三视图，保留尺寸与隐藏线。缺失的尺寸需要补充。</span></div>
        <button className="primary wide" onClick={run} disabled={!job||locked}>{running?<LoaderCircle size={16} className="spin"/>:<Box size={16}/>} {running?'正在处理…':job?.kind==='demo'?'生成参数化样例':'开始建模'} {!running&&<ArrowRight size={15}/>}</button>
        {running&&<button className="text-button cancel" onClick={()=>act(async()=>setJob(await api<Job>(`/jobs/${job!.id}/cancel`,{method:'POST'})))}><Square size={12}/> 取消当前任务</button>}
        </div>
        <div className="sample-card"><div className="sample-icon"><Box size={21}/></div><div><strong>先试一个样例</strong><p>五孔安装板 · 本地真实 CAD</p></div><button aria-label="打开参数化样例" onClick={openDemo} disabled={locked}><ArrowRight size={17}/></button></div>
        <div className="panel-note"><i className="dot"/> {serverConfigured?'已配置服务器，点击建模后发送图纸':'未连接 GPU，上传文件仅保存在本地'}</div>
      </section>

      <section className="panel model-panel"><div className="model-header"><div className="tabs"><button className={tab==='model'?'tab active':'tab'} onClick={()=>setTab('model')}><Box size={15}/> 三维模型</button><button className={tab==='drawing'?'tab active':'tab'} onClick={()=>setTab('drawing')}><FileImage size={15}/> 原始图纸</button><button className={tab==='code'?'tab active':'tab'} onClick={()=>setTab('code')}><Code2 size={15}/> 源码</button></div><div className="version-control">{job?.versions.length?<><select aria-label="模型版本" value={versionNo} onChange={e=>setVersionNo(Number(e.target.value))}><option value={0}>最新版本</option>{job.versions.map(v=><option key={v.number} value={v.number}>v{v.number}</option>)}</select><ChevronDown size={12}/></>:<span className="mono">CAD VIEWPORT</span>}</div></div>
      <div className="canvas-area">{tab==='model'?<ModelViewer url={job&&selected?artifact(job.id,'stl',selected.number):null} busy={locked} onDemo={openDemo}/>:tab==='drawing'?drawingUrl&&(job?.kind==='drawing'||selected)?<div className="drawing-view"><img src={drawingUrl} alt="完整工程图"/><a href={drawingUrl} target="_blank" rel="noreferrer"><Maximize2 size={14}/> 查看原图</a></div>:<div className="simple-empty"><FileImage size={35}/><p>上传图纸后在这里查看</p></div>:selected?<pre className="code-view">{code}</pre>:<div className="simple-empty"><Code2 size={35}/><p>完成建模后查看 CadQuery 源码</p></div>}</div>
      <div className="model-infobar"><div className="model-ident"><span className="model-mini"><Box size={17}/></span><div><strong>{job?.title||'尚未选择零件'}</strong><small>{job?.kind==='demo'?'已知模板 · 非 AI 识图':job?'工程图重建任务':'准备好你的第一张工程图'}</small></div></div><span className={`status-badge ${job?.status==='completed'?'success':''}`}>{job?statusText[job.status]||job.status:'等待输入'}</span></div>
      <div className="model-metrics"><div><span>外形尺寸</span><strong>{selected?selected.report.dimensions.map(n=>Number(n.toFixed(2))).join(' × '):'—'} <small>{selected?'mm':''}</small></strong></div><div><span>实体体积</span><strong>{selected?Math.round(selected.report.volume).toLocaleString():'—'} <small>{selected?'mm³':''}</small></strong></div><div><span>版本</span><strong>{selected?`v${selected.number}`:'—'}</strong></div>{selected&&job&&<a className="stl-link" href={artifact(job.id,'stl',selected.number)}><ArrowDownToLine size={14}/> STL</a>}</div>
      </section>

      <section className="panel agent-panel"><div className="panel-heading"><span><span className="agent-mark">✳</span> 建模助手</span><span className="agent-label">AGENT</span></div>
      <div className="agent-scroll"><div className="agent-intro"><span className="eyebrow">OBSERVE · BUILD · VERIFY</span><p>从输入到实体，<br/>每一步都可以检查。</p></div>
      <div className="section-label">执行过程 <button onClick={()=>setAllEvents(!allEvents)}>{allEvents?'收起':'查看全部'}</button></div>
      <div className="timeline">{displayEvents.length?(allEvents?displayEvents:displayEvents.slice(-5)).map(e=><div className={`timeline-item ${e.level}`} key={e.id}><span className="timeline-icon">{e.level==='success'?<Check size={11}/>:e.level==='error'?<X size={11}/>:e.level==='warning'?<CircleAlert size={11}/>:<Circle size={8}/>}</span><div><p>{e.message}</p><time>{new Date(e.time).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'})}</time></div></div>):['读取工程图','构建 CAD 实体','检查并导出模型'].map((s,i)=><div className="timeline-item inactive" key={s}><span className="timeline-icon">{i+1}</span><p>{s}</p></div>)}</div>
      {selected&&<div className="checks"><div className="section-label">实体检查 <span>{selected.report.checks.filter(c=>c.status==='passed').length} 项通过</span></div>{selected.report.checks.map(c=><div key={c.name} className="check-row" title={c.detail}>{c.status==='passed'?<CheckCircle2 size={14}/>:<CircleAlert size={14}/>}<span>{c.name}</span><small>{c.status==='passed'?'通过':c.status==='failed'?'未通过':'未验证'}</small></div>)}</div>}
      {job?.kind==='demo'&&<div className="parameters"><div className="section-label"><span><SlidersHorizontal size={13}/> 样例参数</span><small>mm</small></div><div className="parameter-grid">{paramFields.map(f=><label key={f.key}>{f.label}<input aria-label={f.label} type="number" min={f.min} max={f.max} value={params[f.key]} disabled={locked} onChange={e=>setParams({...params,[f.key]:Number(e.target.value)})}/></label>)}</div><button className="parameter-apply" onClick={run} disabled={locked}>应用参数并生成新版本 <ArrowRight size={13}/></button></div>}
      <div className="messages">{job?.messages.slice(-6).map((m,i)=><div key={i} className={`message ${m.role}`}>{m.role==='assistant'&&<span className="message-by">建模助手</span>}<p>{m.content}</p></div>)}</div></div>
      <div className="chat-box"><textarea aria-label="对话修改要求" value={draft} onChange={e=>setDraft(e.target.value)} placeholder={job?'描述修改，例如“孔径改为 24 mm”':'先上传图纸或打开样例'} maxLength={2000} disabled={!job||locked} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.nativeEvent.isComposing){e.preventDefault();send()}}}/><div className="chat-bottom"><span>Enter 发送 · Shift + Enter 换行</span><button className="send-button" aria-label="发送修改要求" disabled={!job||locked||!draft.trim()} onClick={send}><ArrowUp size={17}/></button></div></div>
      </section>
    </div>
    <footer className="page-footer"><span><span className="small-dot"/> LOCAL WORKSPACE <span className="footer-separator">/</span> 文件与任务保存在你的设备上</span><span>几何有效 ≠ 图纸还原正确 <span className="footer-separator">·</span> 人工复核尺寸</span></footer>
    </main></div>
    {modal&&<div className="modal-backdrop" onClick={()=>setModal(null)}><section className="modal" role="dialog" aria-modal="true" aria-label={modal==='connection'?'服务器连接':'任务历史'} onClick={e=>e.stopPropagation()}><button className="modal-close" aria-label="关闭" onClick={()=>setModal(null)}><X size={20}/></button>{modal==='connection'?<><span className="modal-icon"><Network size={24}/></span><h2>连接你的建模服务器</h2><p>本地负责文件、任务与预览。GPU 服务器负责真实图纸的模型推理与 CAD 重建。</p><div className="connection-state"><i className={connection?.reachable?'dot green':'dot'}/>{connection?.message||'正在检查连接…'}</div><ol className="setup-list"><li><strong>在服务器部署适配器</strong><span>按项目 docs/REMOTE_API.md 实现 /health 与 /v1/reconstruct。</span></li><li><strong>建立 SSH 隧道</strong><code>ssh -N -L 9000:127.0.0.1:8001 USER@SERVER</code></li><li><strong>配置本地 .env 并重启后端</strong><code>CAD_REMOTE_URL=http://127.0.0.1:9000</code><span>需要认证时，将令牌填入本地 .env；页面不会读取或显示它。</span></li></ol><button className="primary wide" onClick={testConnection}><Network size={15}/> 重新检查连接</button><a className="api-docs" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">生产服务 API 文档位于 http://127.0.0.1:8000/docs ↗</a></>:<><span className="modal-icon"><FolderClock size={24}/></span><h2>任务历史</h2><p>图纸、执行记录和已完成版本保存在本地，刷新后可以继续。</p><div className="history-list">{jobs.length?jobs.map(j=><button key={j.id} onClick={()=>choose(j.id)}><span className="history-file"><FileImage size={18}/></span><span><strong>{j.title}</strong><small>{new Date(j.created_at).toLocaleString('zh-CN')} · {j.version_count} 个版本</small></span><span className="history-status">{statusText[j.status]}</span><ArrowRight size={15}/></button>):<div className="simple-empty"><Clock3 size={28}/><p>还没有任务，先上传一张图纸吧。</p></div>}</div></>}</section></div>}
  </div>
}
