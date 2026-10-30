import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { STLLoader } from 'three/addons/loaders/STLLoader.js'
import { Box, RotateCcw, ScanLine, Grid2X2, LoaderCircle } from 'lucide-react'

export default function ModelViewer({url,busy,onDemo}:{url:string|null;busy:boolean;onDemo:()=>void}){
  const element=useRef<HTMLDivElement>(null)
  const [error,setError]=useState('')
  const [loading,setLoading]=useState(false)
  const [wire,setWire]=useState(false)
  const sceneRef=useRef<{camera:THREE.PerspectiveCamera;controls:OrbitControls;mesh:THREE.Mesh;distance:number}|null>(null)
  useEffect(()=>{if(sceneRef.current)(sceneRef.current.mesh.material as THREE.MeshStandardMaterial).wireframe=wire},[wire])
  useEffect(()=>{
    if(!element.current||!url)return
    const el=element.current
    setError('');setLoading(true)
    let renderer:THREE.WebGLRenderer
    try{renderer=new THREE.WebGLRenderer({antialias:true,alpha:true})}catch{setError('浏览器未能启动 WebGL。请开启硬件加速，或下载 STL 在 CAD 软件中查看。');setLoading(false);return}
    const scene=new THREE.Scene()
    scene.background=new THREE.Color('#f4f3ef')
    const camera=new THREE.PerspectiveCamera(36,1,.1,10000)
    camera.up.set(0,0,1)
    const controls=new OrbitControls(camera,renderer.domElement)
    controls.enableDamping=true;controls.dampingFactor=.08;controls.minDistance=10;controls.maxDistance=2000
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.shadowMap.enabled=true
    renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.outputColorSpace=THREE.SRGBColorSpace
    el.appendChild(renderer.domElement)
    scene.add(new THREE.AmbientLight('#ffffff',1.6))
    const key=new THREE.DirectionalLight('#ffffff',3.2);key.position.set(80,-100,160);key.castShadow=true
    key.shadow.mapSize.set(2048,2048);key.shadow.camera.left=-180;key.shadow.camera.right=180;key.shadow.camera.top=180;key.shadow.camera.bottom=-180
    key.shadow.normalBias=.06;scene.add(key)
    const fill=new THREE.DirectionalLight('#c7c1ff',1.4);fill.position.set(-100,80,40);scene.add(fill)
    let disposed=false,frame=0
    const controller=new AbortController()
    fetch(url,{signal:controller.signal}).then(r=>{if(!r.ok)throw new Error('模型文件尚未就绪');return r.arrayBuffer()}).then(data=>{
      if(disposed)return
      const geometry=new STLLoader().parse(data)
      geometry.computeBoundingBox();const size=new THREE.Vector3();geometry.boundingBox!.getSize(size);geometry.center();geometry.computeVertexNormals()
      const material=new THREE.MeshStandardMaterial({color:'#8b84bb',metalness:.32,roughness:.35})
      const mesh=new THREE.Mesh(geometry,material);mesh.castShadow=true;mesh.receiveShadow=true;scene.add(mesh)
      const edges=new THREE.LineSegments(new THREE.EdgesGeometry(geometry,32),new THREE.LineBasicMaterial({color:'#5a527f',transparent:true,opacity:.5}));mesh.add(edges)
      const span=Math.max(size.x,size.y,size.z)
      const floor=new THREE.Mesh(new THREE.PlaneGeometry(span*8,span*8),new THREE.ShadowMaterial({opacity:.12}));floor.position.z=-size.z/2-span*.002;floor.receiveShadow=true;scene.add(floor)
      const grid=new THREE.GridHelper(span*4,32,'#d1d0cb','#e4e3de');grid.rotation.x=Math.PI/2;grid.position.z=-size.z/2-span*.003;scene.add(grid)
      const distance=span*2.4;controls.minDistance=span*.1;controls.maxDistance=span*100;camera.near=span/1000;camera.far=span*200;camera.updateProjectionMatrix();camera.position.set(distance*.8,-distance,distance*.8)
      controls.target.set(0,0,0);controls.update();sceneRef.current={camera,controls,mesh,distance};setWire(false);setLoading(false)
    }).catch(e=>{if(!disposed&&e.name!=='AbortError'){setError(e.message||'模型预览加载失败');setLoading(false)}})
    const resize=()=>{const w=el.clientWidth,h=el.clientHeight;if(w&&h){renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix()}}
    const ro=new ResizeObserver(resize);ro.observe(el);resize()
    const tick=()=>{frame=requestAnimationFrame(tick);controls.update();renderer.render(scene,camera)};tick()
    return()=>{disposed=true;controller.abort();cancelAnimationFrame(frame);ro.disconnect();controls.dispose();sceneRef.current=null
      scene.traverse(obj=>{const o=obj as THREE.Mesh;if(o.geometry)o.geometry.dispose();if(o.material){const ms=Array.isArray(o.material)?o.material:[o.material];ms.forEach(m=>m.dispose())}})
      renderer.dispose();renderer.domElement.remove()}
  },[url])
  const view=(type:string)=>{const s=sceneRef.current;if(!s)return;const d=s.distance;s.camera.up.set(0,0,1);if(type==='top'){s.camera.position.set(0,0,d*1.5);s.camera.up.set(0,1,0)}else if(type==='front')s.camera.position.set(0,-d*1.5,0);else s.camera.position.set(d*.8,-d,d*.8);s.controls.target.set(0,0,0);s.controls.update()}
  return <div className="viewer"><div className="webgl" ref={element}/>
    {!url&&<div className="viewer-empty"><div className="blueprint-box"><Box size={82} strokeWidth={.65}/><span className="orbit-line"/></div><span className="eyebrow">YOUR NEXT DIMENSION</span><h2>让图纸，成为立体。</h2><p>上传工程图，或先打开一个真实的参数化样例。<br/>在这里查看、检查与迭代你的 CAD 模型。</p><button className="secondary" onClick={onDemo} disabled={busy}><Box size={15}/> 打开安装板样例 <span>↗</span></button></div>}
    {loading&&<div className="viewer-overlay"><LoaderCircle className="spin" size={22}/> 加载实体预览</div>}
    {error&&<div className="viewer-overlay error-text">{error}</div>}
    {url&&<><div className="viewer-caption"><span className="small-dot"/> 实体预览 <span className="mono">PERSPECTIVE</span></div><div className="view-toolbar"><button title="等轴测视角" onClick={()=>view('iso')}><Box size={17}/></button><button title="俯视图" onClick={()=>view('top')}><Grid2X2 size={17}/></button><button title="正视图" onClick={()=>view('front')}><ScanLine size={17}/></button><span/><button title="线框显示" className={wire?'is-on':''} onClick={()=>setWire(!wire)}>⌗</button><button title="重置视角" onClick={()=>view('iso')}><RotateCcw size={16}/></button></div></>}
    <div className="viewer-footer"><span>拖动旋转 · 滚轮缩放 · 右键平移</span><span className="axis"><i>Z</i><b>Y</b><em>X</em></span></div>
  </div>
}
