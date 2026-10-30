export type Params={length:number;width:number;thickness:number;hole:number}
export type Check={name:string;status:string;detail:string}
export type Version={number:number;attempt:string;created_at:string;parameters:Params|null;report:{dimensions:number[];volume:number;unit:string;checks:Check[]}}
export type TaskEvent={id:number;time:string;stage:string;level:string;message:string}
export type Job={id:string;kind:'drawing'|'demo';title:string;status:string;created_at:string;updated_at:string;instructions:string;parameters:Params|null;versions:Version[];messages:{role:string;content:string;time:string}[];events:TaskEvent[];error:string|null;page:number;pages:number}
export type JobSummary=Pick<Job,'id'|'kind'|'title'|'status'|'created_at'|'updated_at'>&{version_count:number}
export type Connection={configured:boolean;reachable:boolean;message:string}
export const statusText:Record<string,string>={ready:'准备就绪',waiting_server:'等待服务器',queued:'排队中',preparing:'准备图纸',generating:'正在建模',checking:'检查实体',completed:'已生成',failed:'处理失败',cancelled:'已取消',interrupted:'已中断'}
export const active=(status:string)=>['queued','preparing','generating','checking'].includes(status)
export async function api<T>(path:string,init?:RequestInit):Promise<T>{
  const response=await fetch('/api'+path,init)
  if(!response.ok){let message='请求失败，请检查本地后端';try{const body=await response.json();message=typeof body.detail==='string'?body.detail:'输入参数不符合要求，请检查数值范围'}catch{/* HTTP fallback */}throw new Error(message)}
  return response.json()
}
export const json=(body:unknown)=>({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
export const artifact=(id:string,key:string,version:number)=>`/api/jobs/${id}/artifacts/${key}?version=${version}`
