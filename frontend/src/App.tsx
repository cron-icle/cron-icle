import { useEffect, useState } from "react";
import * as api from "./api";
type SemanticEvent={id:string;raw_event_id:string;timestamp_ns:number;app_name?:string;window_title?:string;category:string;summary:string;confidence:number;model_name:string;created_at:string}; type RawEvent={id:string;timestamp_ns:number;event_type:string;source:string;app_name?:string;window_title?:string;text?:string;metadata_json:string;privacy_class:string;confidence:number;created_at:string}; type Overview={event:RawEvent;processing:Array<{task_type:string;status:string;attempts:number;error?:string}>;semantic_ready:boolean;embedding_ready:boolean}; type Settings={enabled:boolean;mouse_enabled:boolean;keyboard_enabled:boolean;screenshots_enabled:boolean;keyboard_text_allowlist:string[];excluded_applications:string[];excluded_paths:string[];watched_folders:string[]}; type Queue={pending:number;processing:number;complete:number;failed:number;cancelled:number}; type SetupStatus={runtime_installed:boolean;chat_model_installed:boolean;embed_model_installed:boolean;chat_running:boolean;embed_running:boolean;chat_model_name:string;embed_model_name:string}; type DownloadProgress={label:string;downloaded_bytes:number;total_bytes?:number|null;percent?:number|null}; type MoveProgress={copied_bytes:number;total_bytes:number;percent:number};
const setupComplete=(s:SetupStatus|null)=>!!s&&s.runtime_installed&&s.chat_model_installed&&s.embed_model_installed&&s.chat_running&&s.embed_running;
const formatBytes=(bytes:number)=>{if(!bytes)return"0 B";const units=["B","KB","MB","GB","TB"];const exp=Math.min(Math.floor(Math.log2(bytes)/10),units.length-1);return`${(bytes/2**(10*exp)).toFixed(exp?1:0)} ${units[exp]}`};
const time=(n:number)=>new Date(n/1e6).toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"}); const insights=(q?:string)=>api.listSemanticEvents<SemanticEvent[]>(100,q||null).catch(()=>[]); const raw=()=>api.listRawEventProcessingOverview<Overview[]>(100).catch(()=>[]);
function App(){const [data,setData]=useState<SemanticEvent[]>([]),[rawData,setRawData]=useState<Overview[]>([]),[query,setQuery]=useState(""),[section,setSection]=useState<"timeline"|"search"|"raw"|"settings">("timeline"),[settings,setSettings]=useState<Settings|null>(null),[queue,setQueue]=useState<Queue>({pending:0,processing:0,complete:0,failed:0,cancelled:0}),[startupError,setStartupError]=useState<string|null>(null),[setup,setSetup]=useState<SetupStatus|null>(null); const refresh=()=>{insights(section==="search"?query:undefined).then(setData);if(section==="raw")raw().then(setRawData)}; useEffect(()=>{refresh();api.getCaptureSettings<Settings>().then(setSettings).catch(()=>{});api.processingQueueStatus<Queue>().then(setQueue).catch(()=>{});api.startupDiagnostics<string|null>().then(setStartupError).catch(()=>{});api.localAiSetupStatus<SetupStatus>().then(setSetup).catch(()=>{})},[]);useEffect(()=>{if(section==="search"||section==="raw")refresh()},[query,section]);const capture=async(enabled:boolean)=>{await(enabled?api.startCapture():api.stopCapture());setSettings(s=>s?{...s,enabled}:s)};const permission=async(input:"mouse"|"keyboard",enabled:boolean)=>setSettings(await api.setInputPermission<Settings>(input,enabled));const exportData=async()=>{const link=document.createElement("a");link.href=URL.createObjectURL(new Blob([await api.exportData()],{type:"application/json"}));link.download="cronicle-export.json";link.click()};const del=async()=>{if(confirm("Delete all Cronicle data?")){await api.deleteAllData();setData([]);setRawData([])}};return <main className="app-shell">{startupError&&<div className="banner banner-error" role="alert">{startupError}</div>}{setup&&!setupComplete(setup)&&section!=="settings"&&<div className="banner banner-info" role="status">Local AI setup isn't finished yet — semantic insights won't appear until the local engine and its models are ready. <button className="quiet-button" onClick={()=>setSection("settings")}>Finish setup</button></div>}<aside className="sidebar"><div className="brand"><span className="brand-mark">C</span><span>Cronicle</span></div><nav><button className={`nav-item ${section==="timeline"?"active":""}`} onClick={()=>setSection("timeline")}>Timeline</button><button className={`nav-item ${section==="search"?"active":""}`} onClick={()=>setSection("search")}>Search</button><button className={`nav-item ${section==="raw"?"active":""}`} onClick={()=>setSection("raw")}>Raw Evidence</button><button className={`nav-item ${section==="settings"?"active":""}`} onClick={()=>setSection("settings")}>Settings</button></nav><div className="capture-card"><div className={`status-dot ${settings?.enabled?"on":""}`}/><div><strong>{settings?.enabled?"Capture enabled":"Capture is off"}</strong><span>{queue.pending} processing tasks pending.</span></div></div></aside><section className="content"><header className="topbar"><div><p className="eyebrow">LOCAL MEMORY ENGINE</p><h1>{section==="raw"?"Raw Evidence":section[0].toUpperCase()+section.slice(1)}</h1></div><button className="quiet-button" onClick={refresh}>Refresh</button></header>{section==="settings"?<SettingsPage settings={settings} capture={capture} permission={permission} exportData={exportData} del={del}/>:section==="raw"?<RawPage data={rawData}/>:<><div className="hero"><div><p className="eyebrow">TODAY</p><h2>Your computer, remembered.</h2><p className="muted">Only processed local-model insights appear in the main feed.</p></div><div className="metric"><span>Insights ready</span><strong>{data.length}</strong></div></div>{section==="search"&&<input className="search-input" placeholder="Search processed activity…" value={query} onChange={e=>setQuery(e.target.value)}/>}<section className="timeline"><div className="section-heading"><h3>{section==="search"?"Processed results":"Recent insights"}</h3><span className="muted">LLM processed</span></div>{data.length?data.map(e=><article className="event-row" key={e.id}><time>{time(e.timestamp_ns)}</time><div className="event-icon">•</div><div className="event-body"><div className="event-top"><strong>{e.app_name||"Unknown application"}</strong><span className="event-status">{Math.round(e.confidence*100)}% confidence</span></div><p>{e.summary}</p><span className="event-type">{e.category} · {e.model_name}</span></div></article>):<p className="muted empty">No processed insights yet.</p>}</section>{section==="timeline"&&<button className="quiet-button" onClick={()=>setSection("raw")}>View raw capture evidence</button>}</>}</section></main>}
function LocalAiSetupPanel(){
  const [status,setStatus]=useState<SetupStatus|null>(null);
  const [dataDirConfigured,setDataDirConfigured]=useState(false);
  const [busy,setBusy]=useState<string|null>(null);
  const [progress,setProgress]=useState<number|null>(null);
  const [progressLabel,setProgressLabel]=useState<string|null>(null);
  const [progressBytes,setProgressBytes]=useState<{downloaded:number;total?:number|null}|null>(null);
  const [error,setError]=useState<string|null>(null);
  const refreshStatus=()=>{
    api.localAiSetupStatus<SetupStatus>().then(setStatus).catch(()=>{});
    api.getDataDirectory<string|null>().then(dir=>setDataDirConfigured(!!dir)).catch(()=>{});
  };
  useEffect(()=>{
    refreshStatus();
  },[]);
  useEffect(()=>{
    if(busy!=="setup_download_chat_model"&&busy!=="setup_download_embed_model")return;
    return api.pollProgress<DownloadProgress>(()=>api.localAiDownloadProgress<DownloadProgress>(),p=>{
      if(!p)return;
      setProgressLabel(p.label);
      setProgressBytes({downloaded:p.downloaded_bytes,total:p.total_bytes});
      if(typeof p.percent==="number")setProgress(p.percent);
    });
  },[busy]);
  useEffect(()=>{if(setupComplete(status))return;const id=setInterval(refreshStatus,4000);return()=>clearInterval(id)},[status]);
  const run=async(action:string,task:()=>Promise<unknown>)=>{
    setBusy(action);setError(null);setProgress(null);setProgressLabel(null);setProgressBytes(null);
    try{await task()}
    catch(err){setError(String(err))}
    finally{setBusy(null);setProgress(null);setProgressLabel(null);setProgressBytes(null);await refreshStatus()}
  };
  const artifactActions:Record<string,()=>Promise<void>>={setup_remove_chat_model:api.setupRemoveChatModel,setup_remove_embed_model:api.setupRemoveEmbedModel};
  const removeArtifact=(label:string,command:string)=>{if(confirm(`Remove ${label}? You'll need to download it again to use local AI features that depend on it.`))run(command,artifactActions[command])};
  const cancelDownload=()=>api.cancelModelDownload().catch(()=>{});
  if(!status)return null;
  const ready=setupComplete(status);
  const byteText=progressBytes?(progressBytes.total?`${formatBytes(progressBytes.downloaded)} / ${formatBytes(progressBytes.total)}`:formatBytes(progressBytes.downloaded)):null;
  const progressBar=busy&&(progress!=null||progressLabel)?<div className="setup-progress">
    <div className="setup-progress-track"><div className="setup-progress-fill" style={progress!=null?{width:`${Math.min(100,Math.max(0,progress))}%`}:{width:"100%"}}/></div>
    <span>{progressLabel||"Working…"}{byteText?` — ${byteText}`:""}{progress!=null?` (${Math.round(progress)}%)`:""}</span>
  </div>:null;
  return <div className="setup-panel">
    <h3>Local AI setup</h3>
    <p className="muted">Cronicle analyzes activity entirely on this machine via a bundled llama.cpp engine — no separate app to install, nothing leaves your computer. {ready?"Everything is set up and ready.":"Finish setup once to enable semantic search and insights."}</p>
    {!dataDirConfigured&&<p className="banner banner-info" role="status">Choose a data directory above first — models download into it.</p>}
    <ul className="setup-checklist">
      <li className={status.chat_model_installed?"done":""}>
        <span>Analysis model ({status.chat_model_name})</span>
        <span className="button-row">
          {!status.chat_model_installed&&<button className="quiet-button" disabled={!!busy||!dataDirConfigured} onClick={()=>run("setup_download_chat_model",api.setupDownloadChatModel)}>{busy==="setup_download_chat_model"?"Downloading…":"Download"}</button>}
          {busy==="setup_download_chat_model"&&<button className="quiet-button danger-button" onClick={cancelDownload}>Cancel</button>}
          {status.chat_model_installed&&<button className="quiet-button danger-button" disabled={!!busy} onClick={()=>removeArtifact(status.chat_model_name,"setup_remove_chat_model")}>Remove</button>}
        </span>
      </li>
      <li className={status.embed_model_installed?"done":""}>
        <span>Embedding model ({status.embed_model_name})</span>
        <span className="button-row">
          {!status.embed_model_installed&&<button className="quiet-button" disabled={!!busy||!dataDirConfigured} onClick={()=>run("setup_download_embed_model",api.setupDownloadEmbedModel)}>{busy==="setup_download_embed_model"?"Downloading…":"Download"}</button>}
          {busy==="setup_download_embed_model"&&<button className="quiet-button danger-button" onClick={cancelDownload}>Cancel</button>}
          {status.embed_model_installed&&<button className="quiet-button danger-button" disabled={!!busy} onClick={()=>removeArtifact(status.embed_model_name,"setup_remove_embed_model")}>Remove</button>}
        </span>
      </li>
      <li className={status.chat_running&&status.embed_running?"done":""}>
        <span>Engine running</span>
        {status.chat_model_installed&&status.embed_model_installed&&(!status.chat_running||!status.embed_running)&&<button className="quiet-button" disabled={!!busy} onClick={()=>run("setup_start_engine",api.setupStartEngine)}>{busy==="setup_start_engine"?"Starting…":"Start"}</button>}
      </li>
    </ul>
    {progressBar}
    {error&&<p className="banner banner-error">{error}</p>}
  </div>;
}
function DataDirectoryPanel(){
  const [path,setPath]=useState<string|null|undefined>(undefined);
  const [busy,setBusy]=useState(false);
  const [moveProgress,setMoveProgress]=useState<MoveProgress|null>(null);
  const [error,setError]=useState<string|null>(null);
  const configured=!!path;
  const refreshPath=()=>api.getDataDirectory<string|null>().then(setPath).catch(()=>{});
  useEffect(()=>{refreshPath()},[]);
  useEffect(()=>{
    if(!busy)return;
    return api.pollProgress<MoveProgress>(()=>api.dataDirectoryMoveProgress<MoveProgress>(),setMoveProgress);
  },[busy]);
  const change=async()=>{
    const prompt=configured
      ?"Cronicle will check the chosen folder has enough free space, move all of its data there, then restart. Continue?"
      :"Choose a folder for Cronicle to store its data and downloaded models. Cronicle will restart once it's set.";
    if(!confirm(prompt))return;
    setBusy(true);setError(null);setMoveProgress(null);
    try{await api.changeDataDirectory()}
    catch(err){setError(String(err));setBusy(false);setMoveProgress(null)}
  };
  const moveProgressBar=busy&&moveProgress?<div className="setup-progress">
    <div className="setup-progress-track"><div className="setup-progress-fill" style={{width:`${Math.min(100,Math.max(0,moveProgress.percent))}%`}}/></div>
    <span>Moving data — {formatBytes(moveProgress.copied_bytes)} / {formatBytes(moveProgress.total_bytes)} ({Math.round(moveProgress.percent)}%)</span>
  </div>:null;
  return <div className="setup-panel">
    <h3>Data directory</h3>
    <p className="muted">Everything Cronicle stores — the event database and downloaded models — lives in this folder. {configured?"":"Not set yet — Cronicle is running in a temporary, non-persistent mode until you choose one."}</p>
    <p className="setting-row"><span><strong>Current location</strong><small>{path===undefined?"Loading…":path||"Not set"}</small></span><button className="quiet-button" disabled={busy} onClick={change}>{busy?"Moving data…":configured?"Change directory…":"Choose directory…"}</button></p>
    {moveProgressBar}
    {busy&&!moveProgress&&<p className="muted">Checking free space and preparing…</p>}
    {error&&<p className="banner banner-error">{error}</p>}
  </div>;
}
function SettingsPage({settings,capture,permission,exportData,del}:{settings:Settings|null;capture:(v:boolean)=>void;permission:(i:"mouse"|"keyboard",v:boolean)=>void;exportData:()=>void;del:()=>void}){const [excludedApps,setExcludedApps]=useState("");const [excludedPaths,setExcludedPaths]=useState("");const [watched,setWatched]=useState("");const [screenshotsEnabled,setScreenshotsEnabled]=useState(false);const [keyboardAllowlist,setKeyboardAllowlist]=useState("");useEffect(()=>{setExcludedApps(settings?.excluded_applications.join("\n")||"");setExcludedPaths(settings?.excluded_paths?.join("\n")||"");setWatched(settings?.watched_folders.join("\n")||"");setScreenshotsEnabled(settings?.screenshots_enabled||false);setKeyboardAllowlist(settings?.keyboard_text_allowlist?.join("\n")||"")},[settings]);const saveExcludedApps=async()=>{const values=excludedApps.split(/[\n,]/).map(v=>v.trim()).filter(Boolean);const next=await api.setExcludedApplications<Settings>(values);setExcludedApps(next.excluded_applications.join("\n"))};const saveExcludedPaths=async()=>{const values=excludedPaths.split(/[\n,]/).map(v=>v.trim()).filter(Boolean);const next=await api.setExcludedPaths<Settings>(values);setExcludedPaths((next.excluded_paths||[]).join("\n"))};const saveKeyboardAllowlist=async()=>{const applications=keyboardAllowlist.split(/[\n,]/).map(v=>v.trim()).filter(Boolean);const next=await api.setKeyboardTextAllowlist<Settings>(applications);setKeyboardAllowlist(next.keyboard_text_allowlist.join("\n"))};const saveWatched=async()=>{const values=watched.split(/[\n,]/).map(v=>v.trim()).filter(Boolean);const next=await api.setWatchedFolders<Settings>(values);setWatched(next.watched_folders.join("\n"))};return <section className="settings-panel"><DataDirectoryPanel/><LocalAiSetupPanel/><h2>Permissions and privacy</h2><p className="muted">Raw evidence is retained privately for processing. The main feed shows only semantic insights.</p>{settings&&<><label className="setting-row"><span><strong>Foreground application tracking</strong><small>Records active application and window titles.</small></span><input type="checkbox" checked={settings.enabled} onChange={e=>capture(e.target.checked)}/></label><label className="setting-row"><span><strong>Mouse capture</strong><small>Clicks, scrolling, and drag metadata.</small></span><input type="checkbox" checked={settings.mouse_enabled} onChange={e=>permission("mouse",e.target.checked)}/></label><label className="setting-row"><span><strong>Keyboard metadata</strong><small>Key codes only; text remains protected.</small></span><input type="checkbox" checked={settings.keyboard_enabled} onChange={e=>permission("keyboard",e.target.checked)}/></label><label className="setting-row"><span><strong>Screen capture</strong><small>Transient screenshots after meaningful events.</small></span><input type="checkbox" checked={screenshotsEnabled} onChange={async e=>{const next=await api.setScreenshotPermission<Settings>(e.target.checked);setScreenshotsEnabled(next.screenshots_enabled)}}/></label><div className="exclusion-editor"><strong>Keyboard text allowlist</strong><small>Text capture remains off for every application not listed here.</small><textarea value={keyboardAllowlist} onChange={e=>setKeyboardAllowlist(e.target.value)} placeholder="code.exe\nnotes.exe"/><button className="quiet-button" onClick={saveKeyboardAllowlist}>Save keyboard allowlist</button></div><div className="exclusion-editor"><strong>Excluded applications</strong><small>One executable name per line (exact match, e.g. chrome.exe).</small><textarea value={excludedApps} onChange={e=>setExcludedApps(e.target.value)} placeholder="chrome.exe\npassword-manager.exe"/><button className="quiet-button" onClick={saveExcludedApps}>Save excluded applications</button></div><div className="exclusion-editor"><strong>Excluded paths</strong><small>One folder or path segment per line.</small><textarea value={excludedPaths} onChange={e=>setExcludedPaths(e.target.value)} placeholder="Secrets\nnode_modules"/><button className="quiet-button" onClick={saveExcludedPaths}>Save excluded paths</button></div><div className="exclusion-editor"><strong>Watched folders</strong><small>One existing folder path per line.</small><textarea value={watched} onChange={e=>setWatched(e.target.value)} placeholder="E:\\Projects\\Notes"/><button className="quiet-button" onClick={saveWatched}>Save watched folders</button></div><div className="button-row"><button className="quiet-button" onClick={exportData}>Export JSON</button><button className="quiet-button danger-button" onClick={del}>Delete all data</button></div></>}</section>}
function RawPage({data}:{data:Overview[]}){const retry=async()=>{await api.retryFailedProcessingTasks()};return <section className="timeline"><div className="section-heading"><div><h3>Raw capture evidence</h3><p className="muted">Diagnostic source records and processing state.</p></div><div className="button-row"><span className="muted">Internal evidence</span><button className="quiet-button" onClick={retry}>Retry failed processing</button></div></div>{data.length?data.map(o=>{const e=o.event,l=o.processing[o.processing.length-1];return <article className="event-row" key={e.id}><time>{time(e.timestamp_ns)}</time><div className="event-icon">•</div><div className="event-body"><div className="event-top"><strong>{e.app_name||"Unknown application"}</strong><span className="event-status">{l?.status||"captured"}</span></div><p>{e.window_title||e.text||"Activity recorded"}</p><span className="event-type">{e.event_type} · {e.source}</span><small className="muted">Semantic: {o.semantic_ready?"ready":"not ready"} · Embedding: {o.embedding_ready?"ready":"not ready"}{l?` · attempt ${l.attempts}${l.error?` · ${l.error}`:""}`:""}</small></div></article>}):<p className="muted empty">No raw evidence captured.</p>}</section>}
export default App;
