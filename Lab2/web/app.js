// GestureAI course starter: first run works without a custom classifier.
// For production deploy, pin and locally vendor dependencies after instructor preflight.
import { HandLandmarker, FilesetResolver, DrawingUtils } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/+esm';
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/+esm';
const $ = id => document.getElementById(id);
const CLASSES = ['open_palm', 'fist', 'peace', 'thumbs_up'];
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const MODEL_URL = './models/gesture_model.onnx';
const LANDMARK_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm';
let detector, session, videoStream, active=false, capturing=false, rows=[], frames=0, lastFPS=performance.now(), lastVideoTime=-1;
let classifierClasses=CLASSES, stable='', streak=0, lastActionAt=0, slide=0, paused=false;
const slides=['Welcome to GestureAI','All AI inference runs locally','Trained with your own landmark dataset','Exported from PyTorch to ONNX','Measured and optimized for the edge'];
const status = msg => { $('status').textContent=msg; };
function features(landmarks){
  // The EXACT same 42-feature transform is used in Python notebooks.
  const w=landmarks[0]; let xy=landmarks.flatMap(p=>[p.x-w.x,p.y-w.y]);
  const maxRadius=Math.max(1e-6,...Array.from({length:21},(_,i)=>Math.hypot(xy[i*2],xy[i*2+1])));
  return xy.map(v=>v/maxRadius);
}
function softmax(logits){const m=Math.max(...logits), p=logits.map(x=>Math.exp(x-m));const z=p.reduce((a,b)=>a+b,0);return p.map(x=>x/z);}
function act(gesture){
  if(paused && gesture!=='peace')return;
  if(gesture==='open_palm') slide=(slide+1)%slides.length;
  if(gesture==='fist') slide=(slide-1+slides.length)%slides.length;
  if(gesture==='peace') paused=!paused;
  if(gesture==='thumbs_up') $('action').textContent='Confirmed ✓';
  if(gesture!=='thumbs_up') $('action').textContent=`Action: ${gesture}`+(paused?' (paused)':'');
  $('slide').textContent=`Slide ${slide+1} · ${slides[slide]}`;
}
async function startCamera(){
 try {
  if(active)return;
  status('Loading hand landmark model...');
  if(!detector){const vision=await FilesetResolver.forVisionTasks(WASM_URL);detector=await HandLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:LANDMARK_URL,delegate:'CPU'},runningMode:'VIDEO',numHands:1,minHandDetectionConfidence:.5});}
  videoStream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:false});
  $('video').srcObject=videoStream;await $('video').play();active=true;lastVideoTime=-1;
  const canvas=$('overlay');canvas.width=$('video').videoWidth;canvas.height=$('video').videoHeight;
  status('Camera ready. Hand skeleton detected locally. Load classifier after Lab 3.');requestAnimationFrame(loop);
 }catch(e){status('Camera/model error: '+e.message+' (use localhost or HTTPS; allow camera permission).');}
}
async function loadClassifier(){
 try{status('Loading student-trained ONNX classifier...');const resp=await fetch('./models/classes.json');if(resp.ok)classifierClasses=await resp.json();
   session=await ort.InferenceSession.create(MODEL_URL,{executionProviders:['wasm']});status('Classifier loaded: '+classifierClasses.join(', '));
 }catch(e){status('No compatible ONNX model yet: '+e.message+'. Continue Lab 1/2 with landmarks only.');}
}
async function classify(x){const t0=performance.now();const input=new ort.Tensor('float32',Float32Array.from(x),[1,42]);const result=await session.run({[session.inputNames[0]]:input});const logits=Array.from(result[session.outputNames[0]].data);const probs=softmax(logits);const k=probs.indexOf(Math.max(...probs));return {label:classifierClasses[k]||String(k),confidence:probs[k],ms:performance.now()-t0};}
async function handleFeatures(x){
 if(session){const result=await classify(x);$('classTime').textContent=result.ms.toFixed(2)+' ms';$('confidence').textContent=(100*result.confidence).toFixed(1)+'%';$('prediction').textContent=result.label;
    // Lab 6: students modify streak threshold and cooldown; no cloud inference.
    if(result.confidence>=.75){if(result.label===stable)streak++;else{stable=result.label;streak=1;}if(streak>=5 && Date.now()-lastActionAt>1600){act(result.label);lastActionAt=Date.now();}}
    else{stable='';streak=0;}
 }else $('prediction').textContent='Hand detected (load your model in Lab 3)';
}
async function loop(){
 if(!active)return;
 try{
  const video=$('video'),canvas=$('overlay'),ctx=canvas.getContext('2d');
  if(video.readyState>=2 && video.currentTime!==lastVideoTime){
   lastVideoTime=video.currentTime;const t0=performance.now();const res=detector.detectForVideo(video,performance.now());
   $('landmarkTime').textContent=(performance.now()-t0).toFixed(1)+' ms';ctx.clearRect(0,0,canvas.width,canvas.height);
   if(res.landmarks?.length){
    const points=res.landmarks[0];const draw=new DrawingUtils(ctx);draw.drawConnectors(points,HandLandmarker.HAND_CONNECTIONS,{color:'#54ffcb',lineWidth:3});draw.drawLandmarks(points,{color:'#ffbe55',radius:3});
    const x=features(points);
    if(capturing && Date.now()-window.lastSaved>180){rows.push([$('label').value,$('session').value,Date.now(),...x]);window.lastSaved=Date.now();$('count').textContent=rows.length+' samples (features only; no video saved)';}
    await handleFeatures(x);
   }else{$('prediction').textContent='No hand';$('confidence').textContent='—';stable='';streak=0;}
   frames++;if(performance.now()-lastFPS>1000){$('fps').textContent=(frames*1000/(performance.now()-lastFPS)).toFixed(1);lastFPS=performance.now();frames=0;}
  }
 }catch(e){status('Processing error: '+e.message);}
 if(active)requestAnimationFrame(loop);
}
$('start').onclick=startCamera;
$('stop').onclick=()=>{active=false;replaying=false;capturing=false;$('capture').textContent='Start Capture';videoStream?.getTracks().forEach(t=>t.stop());$('video').srcObject=null;status('Camera stopped.');};
$('load').onclick=loadClassifier;
$('capture').onclick=()=>{capturing=!capturing;window.lastSaved=0;$('capture').textContent=capturing?'Stop Capture':'Start Capture';};
$('clear').onclick=()=>{rows=[];$('count').textContent='0 samples (features only; no video saved)';};
$('export').onclick=()=>{if(!rows.length)return alert('Collect samples first.');const head=['label','session','timestamp',...Array.from({length:42},(_,i)=>'f'+i)];const csv=[head.join(','),...rows.map(r=>r.join(','))].join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='gesture_'+$('session').value+'.csv';a.click();URL.revokeObjectURL(a.href);};
// No-webcam mode: replay a recorded feature CSV (Lab 2 format) at its recorded timing through the same classifier and decision logic.
let replaying=false;
function drawFeatures(x){
  const canvas=$('overlay'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  const s=canvas.height*0.32,cx=canvas.width/2,cy=canvas.height*0.72,p=i=>[cx+x[2*i]*s,cy+x[2*i+1]*s];
  ctx.strokeStyle='#54ffcb';ctx.lineWidth=3;
  for(const {start,end} of HandLandmarker.HAND_CONNECTIONS){ctx.beginPath();ctx.moveTo(...p(start));ctx.lineTo(...p(end));ctx.stroke();}
  ctx.fillStyle='#ffbe55';for(let i=0;i<21;i++){ctx.beginPath();ctx.arc(...p(i),3,0,2*Math.PI);ctx.fill();}
}
async function replayCSV(text){
  const lines=text.trim().split(/\r?\n/),head=lines[0].split(',');
  const fi=Array.from({length:42},(_,i)=>head.indexOf('f'+i)),ti=head.indexOf('timestamp');
  if(fi.some(i=>i<0)||ti<0)throw new Error('CSV needs timestamp and f0..f41 columns (Lab 2 format).');
  if(!session)await loadClassifier();
  $('stop').click();replaying=true;const canvas=$('overlay');canvas.width=640;canvas.height=480;
  status('Replaying recorded features (no camera).');let prev=null;
  for(const line of lines.slice(1)){
    if(!replaying)break;
    const c=line.split(','),t=+c[ti];
    // A jump between recordings is replayed as the hand leaving the view.
    if(prev!==null&&t-prev>300){stable='';streak=0;$('prediction').textContent='No hand';canvas.getContext('2d').clearRect(0,0,640,480);await new Promise(r=>setTimeout(r,300));}
    else if(prev!==null)await new Promise(r=>setTimeout(r,t-prev));
    prev=t;const x=fi.map(i=>+c[i]);drawFeatures(x);await handleFeatures(x);
    frames++;if(performance.now()-lastFPS>1000){$('fps').textContent=(frames*1000/(performance.now()-lastFPS)).toFixed(1);lastFPS=performance.now();frames=0;}
  }
  status(replaying?'Replay finished.':'Replay stopped.');replaying=false;
}
window.replayCSV=replayCSV;
$('replay').onclick=async()=>{const f=$('replayFile').files[0];if(!f)return alert('Choose a feature CSV first (Lab 2 format).');try{await replayCSV(await f.text());}catch(e){status('Replay error: '+e.message);}};
