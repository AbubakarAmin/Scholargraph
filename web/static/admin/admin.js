const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const api=async(path,opts)=>{const r=await fetch(path,opts);const data=await r.json().catch(()=>({}));if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:data.detail?.[0]?.msg||r.statusText);return data};
const short=(s,n=14)=>{const t=String(s??'');return t.length<=n?t:t.slice(0,n)+'…'};
const titleCase=s=>String(s||'').replaceAll('_',' ');

let dashboard={},status={},scratch={},selectedRunId=null,toastTimer=null;

const phases=['topic_discovery','hypothesis_debate','planning','data_validation','writing_narrative','engineering','independent_validation','writing_results','supervision','meta_evaluation','editing'];
const titles={overview:'Run overview',experiments:'Experiments',evidence:'Evidence and lineage',agents:'Agent registry',paper:'Manuscript workspace',activity:'Activity',runs:'Run history',settings:'System settings'};

const settingDefs=[
 ['llm_provider','LLM provider','provider','LLM'],['gemini_model','Gemini model','text','LLM'],['gemini_embedding_model','Gemini embedding model','text','LLM'],['openai_model','OpenAI model','text','LLM'],['openai_embedding_model','OpenAI embedding model','text','LLM'],['openai_base_url','OpenAI-compatible base URL','text','LLM'],['llm_model_cheap','Cheap model override','text','LLM'],['llm_model_strong','Strong model override','text','LLM'],['llm_model_judge','Judge model override','text','LLM'],['ensemble_judge_models','Ensemble judge models','text','LLM'],
 ['research_domain','Research domain','text','Research'],['openalex_email','OpenAlex contact email','email','Research'],['supervisor_threshold','Supervisor threshold','number','Research'],['debate_pass_threshold','Debate pass threshold','number','Research'],['debate_min_rounds','Debate minimum rounds','number','Research'],['debate_max_rounds','Debate maximum rounds','number','Research'],['novelty_similarity_reject','Novelty rejection threshold','number','Research'],['experiment_seeds','Experiment seeds','number','Research'],['experiment_branch_count','Experiment branch count','number','Research'],['max_iterations','Maximum iterations','number','Research'],
 ['debug_mode','Debug mode','checkbox','Runtime'],['log_level','Log level','text','Runtime'],['memory_size','Memory size','number','Runtime'],['sandbox_timeout_sec','Sandbox timeout (seconds)','number','Runtime'],['sandbox_max_output_bytes','Sandbox max output bytes','number','Runtime'],['web_host','Web host','text','Runtime'],['web_port','Web port','number','Runtime'],
 ['vector_db_path','Vector database path','text','Storage'],['cross_run_memory_path','Cross-run memory path','text','Storage'],['run_log_path','Run scratchpad path','text','Storage'],['run_events_path','Run events path','text','Storage'],['elo_ratings_path','Elo ratings path','text','Storage'],['checkpoint_path','Checkpoint path','text','Storage'],['research_db_path','Research ledger path','text','Storage'],['output_dir','Output directory','text','Storage'],['draft_versions_dir','Draft versions directory','text','Storage'],['debate_log_path','Debate log path','text','Storage'],['feedback_log_path','Feedback log path','text','Storage'],['raw_results_dir','Raw results directory','text','Storage'],['companion_repo_dir','Companion repo directory','text','Storage'],['keys_store_path','Keys store path','text','Storage'],
 ['GOOGLE_API_KEY','Google API key','password','Credentials'],['OPENAI_API_KEY','OpenAI API key','password','Credentials'],['SEMANTIC_SCHOLAR_API_KEY','Semantic Scholar API key','password','Credentials'],['SCITE_API_KEY','SCITE API key','password','Credentials']
];
let activeSettingsGroup='LLM';

function toast(msg){
  const el=$('toast');el.textContent=msg;el.classList.add('show');
  clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('show'),1600);
}

async function copyText(text,btn){
  const value=String(text??'');
  if(!value){toast('Nothing to copy');return}
  try{await navigator.clipboard.writeText(value)}catch{
    const ta=document.createElement('textarea');ta.value=value;document.body.appendChild(ta);ta.select();document.execCommand('copy');ta.remove();
  }
  if(btn){btn.classList.add('copied');btn.textContent='copied';setTimeout(()=>{btn.classList.remove('copied');btn.textContent='copy'},900)}
  toast('Copied');
}

function copyBtn(value,label='copy'){
  return `<button type="button" class="copy" data-copy="${esc(value)}">${label}</button>`;
}

function copyRow(value,empty='—'){
  const v=value==null||value===''?'' : String(value);
  if(!v) return `<span class="mono">${empty}</span>`;
  return `<div class="copy-row"><span class="mono" title="${esc(v)}">${esc(short(v,42))}</span>${copyBtn(v)}</div>`;
}

function tagFor(status){
  const s=String(status||'').toLowerCase();
  if(['completed','passed','verified','ready','success','good'].includes(s)) return 'good';
  if(['failed','error','blocked','terminal'].includes(s)) return 'bad';
  if(['running','active','live'].includes(s)) return 'live';
  if(['pending','incomplete','warn','waiting'].includes(s)) return 'warn';
  return 'idle';
}

function workspace(){return dashboard.workspace||{}}

function experimentIndex(){
  const w=workspace();
  const planned=(w.plan?.experiments||[]).filter(x=>x&&typeof x==='object');
  const contracts=w.experiment_contracts||{};
  const exec=w.execution_artifacts||{};
  const eng=w.engineer_outputs||{};
  const analysis=w.analysis_reports||{};
  const names=new Set([
    ...planned.map(x=>x.name).filter(Boolean),
    ...Object.keys(contracts),
    ...Object.keys(exec),
    ...Object.keys(eng),
    ...Object.keys(analysis),
  ]);
  return [...names].map(name=>{
    const spec=planned.find(x=>x.name===name)||null;
    const contract=contracts[name]||null;
    const artifact=exec[name]||null;
    const output=eng[name]||null;
    const report=analysis[name]||null;
    const listed=!!(spec||contract);
    const ran=!!(artifact||output);
    const done=String(artifact?.status||'').toLowerCase()==='completed' || output?.success===true || !!report;
    const failed=String(artifact?.status||'').toLowerCase()==='failed' || output?.success===false;
    let status='planned';
    if(failed) status='failed';
    else if(done) status='completed';
    else if(ran) status='ran';
    else if(listed) status='listed';
    else status='unknown';
    return {name,spec,contract,artifact,output,report,listed,ran,done,failed,status};
  });
}

function renderSettingsForm(values={},keys={}){
  const groups=[...new Set(settingDefs.map(x=>x[3]))];
  $('settingsTabs').innerHTML=groups.map(g=>`<button class="${g===activeSettingsGroup?'primary':''}" data-settings-group="${g}">${g}</button>`).join('');
  document.querySelectorAll('[data-settings-group]').forEach(b=>b.onclick=()=>{activeSettingsGroup=b.dataset.settingsGroup;renderSettingsForm(values,keys)});
  $('settingsForm').innerHTML=settingDefs.map(([id,label,type,group])=>{
    const raw=id===id.toUpperCase()?keys[id]:values[id];
    const value=raw==null?'':raw;
    return `<div data-settings-field="${group}" style="display:${group===activeSettingsGroup?'block':'none'}"><label for="setting_${id}">${label}</label>${
      type==='provider'?`<select id="setting_${id}"><option value="gemini">Gemini</option><option value="openai">OpenAI</option><option value="openai_compatible">OpenAI-compatible</option></select>`:
      type==='checkbox'?`<input id="setting_${id}" type="checkbox" ${value?'checked':''}>`:
      `<input id="setting_${id}" type="${type}" value="${esc(value)}" autocomplete="off">`
    }</div>`;
  }).join('');
  settingDefs.forEach(([id])=>{const el=$(`setting_${id}`);if(el&&id==='llm_provider')el.value=values[id]||'gemini'});
}

function settingValue(id){const el=$(`setting_${id}`);return el?.type==='checkbox'?el.checked:el?.value||''}

async function loadSettings(){
  const[c,k]=await Promise.all([api('/api/config'),api('/api/keys')]);
  renderSettingsForm(c,k.keys||{});
  window.settingsValues=c;window.settingsKeys=k.keys||{};
}

function showView(name){
  document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id===name));
  document.querySelectorAll('.nav button').forEach(x=>x.classList.toggle('active',x.dataset.view===name));
  $('viewTitle').textContent=titles[name]||name;
}

function renderOverview(){
  const w=workspace();
  const findings=w.verification_findings||[];
  const blocking=findings.filter(x=>x.blocking);
  const exps=experimentIndex();
  const doneCount=exps.filter(x=>x.done).length;
  const listedCount=exps.filter(x=>x.listed||x.ran).length||exps.length;
  const ready=!blocking.length && w.reproducibility?.passed && Object.keys(w.execution_artifacts||{}).length>0;
  const terminal=dashboard.release?.status==='blocked'||w.terminal_error||w.evidence_gate?.terminal;

  $('readiness').textContent=terminal?'Blocked':ready?'Ready':'Blocked';
  $('readiness').className='metric-value '+(terminal?'bad':ready?'good':'bad');
  $('readinessNote').textContent=terminal?(dashboard.release?.reason||w.terminal_error||'Terminal evidence failure'):(ready?'Evidence gate passed':'Verification or reproducibility incomplete');

  $('phaseMetric').textContent=titleCase(dashboard.phase||'idle');
  $('phaseNote').textContent=dashboard.status||'idle';
  $('experimentMetric').textContent=`${doneCount} / ${listedCount||0}`;
  $('experimentNote').textContent=listedCount?`${doneCount} completed · ${exps.filter(x=>x.ran&&!x.done).length} in progress · ${exps.filter(x=>x.listed&&!x.ran).length} waiting`:'Awaiting plan experiments';
  $('findingMetric').textContent=blocking.length;
  $('findingMetric').className='metric-value '+(blocking.length?'bad':'good');
  $('topicLine').textContent=w.plan?.title||w.paper?.topic?.title||'No active research run';

  $('runStatus').textContent=status.running?'Running':status.error?'Error':dashboard.status||'Idle';
  $('runId').textContent=dashboard.run_id?'#'+dashboard.run_id:'';
  $('copyRunId').style.display=dashboard.run_id?'inline-flex':'none';
  $('pulse').className='pulse '+(status.running?'live':status.error?'bad':'');
  $('phaseTag').textContent=titleCase(dashboard.phase||'idle');

  const at=phases.indexOf(dashboard.phase);
  $('phaseList').innerHTML=phases.map((p,i)=>`<div class="phase ${p===dashboard.phase?'active':i<at?'done':''}">${titleCase(p)}</div>`).join('');

  const checks=[
    ['Plan exists',!!w.plan],
    ['Data validated',w.data_validation?.passed!==false && !!Object.keys(w.data_artifacts||{}).length],
    ['Execution artifacts',Object.keys(w.execution_artifacts||{}).length>0],
    ['Independent analysis',Object.keys(w.analysis_reports||{}).length>0],
    ['No blocking findings',!blocking.length],
    ['Reproducibility dossier',!!w.reproducibility?.passed],
  ];
  $('gateRows').innerHTML=checks.map(([label,ok])=>`<div class="row"><div><strong>${esc(label)}</strong><small>${ok?'Requirement satisfied':'Evidence still required'}</small></div><span class="tag ${ok?'good':'bad'}">${ok?'PASS':'BLOCKED'}</span></div>`).join('');
  $('gateTag').textContent=terminal?'Blocked':ready?'Ready for editing':'Incomplete';
  $('gateTag').className='tag '+(terminal?'bad':ready?'good':'warn');

  $('expProgress').innerHTML=[
    ['Listed',exps.filter(x=>x.listed||x.ran).length||exps.length],
    ['Ran',exps.filter(x=>x.ran).length],
    ['Done',doneCount],
    ['Failed',exps.filter(x=>x.failed).length],
  ].map(([l,n])=>`<div class="chip"><span class="n">${n}</span><span class="l">${l}</span></div>`).join('');

  $('expOverviewList').innerHTML=exps.length?exps.slice(0,6).map(x=>`<div class="row"><div><strong>${esc(x.name)}</strong><small>${esc(x.spec?.falsifiable_prediction||x.contract?.hypothesis||x.spec?.description||'No prediction recorded')}</small></div><span class="tag ${tagFor(x.status)}">${esc(x.status)}</span></div>`).join('')+(exps.length>6?`<p class="note">${exps.length-6} more on Experiments</p>`:'') : '<div class="empty">No experiments listed yet.</div>';

  const ev=(dashboard.events_tail||[]).slice().reverse();
  $('activity').innerHTML=ev.length?ev.slice(0,12).map(e=>`<div class="event"><time>${esc((e.ts||'').slice(11,19))}</time><strong>${esc(e.agent||'system')}</strong><br><span class="mono">${esc(e.type)}</span></div>`).join(''):'<div class="empty">No activity yet.</div>';

  const scores=w.supervisor_scores||{};
  const scoreVals=Object.values(scores).filter(v=>typeof v==='number');
  const avg=scoreVals.length?(scoreVals.reduce((a,b)=>a+b,0)/scoreVals.length).toFixed(1):'—';
  const debate= (w.debates||[]).at(-1);
  $('snapshotRows').innerHTML=[
    ['Provider',dashboard.config?.provider||'—'],
    ['Model',dashboard.config?.model||'—'],
    ['Domain',dashboard.config?.domain||'—'],
    ['Datasets',Object.keys(w.data_artifacts||{}).length],
    ['Claims', (dashboard.evidence_trace||[]).length],
    ['Supervisor avg',avg],
    ['Debate rounds',debate?.rounds?.length??'—'],
    ['Human approved',w.human_approved?'yes':'no'],
  ].map(([k,v])=>`<div class="row"><div><strong>${esc(k)}</strong></div><span class="mono">${esc(v)}</span></div>`).join('');
}

function artifactCard(kind,name,obj){
  const hash=obj.content_hash||obj.contract_hash||'';
  const loc=obj.location||obj.raw_results_path||'';
  const status=obj.status||(obj.validation?.passed?'validated':'recorded');
  return `<div class="card" style="box-shadow:none;margin-bottom:10px;padding:14px">
    <div class="card-head" style="margin-bottom:8px"><h3 style="font-size:13px">${esc(kind)} · ${esc(name)}</h3><span class="tag ${tagFor(status)}">${esc(status)}</span></div>
    <div class="kv"><span class="k">Hash</span><span class="v mono" title="${esc(hash)}">${esc(hash?short(hash,28):'—')}</span>${hash?copyBtn(hash):'<span></span>'}</div>
    <div class="kv"><span class="k">Path</span><span class="v mono" title="${esc(loc)}">${esc(loc?short(loc,36):'—')}</span>${loc?copyBtn(loc):'<span></span>'}</div>
  </div>`;
}

function renderEvidence(){
  const w=workspace();
  const a=w.data_artifacts||{}, e=w.execution_artifacts||{};
  $('findingTag').textContent=(w.verification_findings||[]).length;
  $('findings').innerHTML=(w.verification_findings||[]).map(f=>`<div class="finding ${f.blocking?'':'ok'}"><strong>${esc(f.check||'finding')}</strong><br>${esc(f.message)}</div>`).join('')||'<div class="finding ok"><strong>No findings</strong><br>Independent verifier has no blockers.</div>';

  const rows=[
    ...Object.entries(a).map(([n,x])=>`<tr><td>Dataset<br><strong>${esc(n)}</strong></td><td>${esc(x.validation?.passed?'validated':'needs review')}</td><td>${copyRow(x.content_hash)}</td><td>${copyRow(x.location)}</td></tr>`),
    ...Object.entries(e).map(([n,x])=>`<tr><td>Execution<br><strong>${esc(n)}</strong></td><td>${esc(x.status||'—')}</td><td>${copyRow(x.content_hash)}</td><td>${copyRow(x.raw_results_path)}</td></tr>`),
  ];
  $('artifactTable').innerHTML=rows.length?`<table><thead><tr><th>Artifact</th><th>Status</th><th>Hash</th><th>Location</th></tr></thead><tbody>${rows.join('')}</tbody></table>`:'<div class="empty">No artifacts recorded.</div>';

  $('datasetCards').innerHTML=Object.keys(a).length?Object.entries(a).map(([n,x])=>artifactCard('Dataset',n,x)).join(''):'<div class="empty">No dataset artifacts.</div>';
  $('execCards').innerHTML=Object.keys(e).length?Object.entries(e).map(([n,x])=>artifactCard('Execution',n,x)).join(''):'<div class="empty">No execution artifacts.</div>';

  $('claims').innerHTML=(dashboard.evidence_trace||[]).map(c=>`<div class="row"><div><strong>${esc(c.claim)}</strong><small>${esc(c.section)} · ${esc(c.type)}</small></div><span class="tag ${c.status==='verified'?'good':'bad'}">${esc(c.status)}</span></div>`).join('')||'<div class="empty">No claim records yet.</div>';
}

function metricsBlock(metrics){
  const entries=Object.entries(metrics||{});
  if(!entries.length) return '<p class="note">No aggregate metrics yet.</p>';
  return entries.map(([k,v])=>{
    if(v&&typeof v==='object'&&('mean' in v||'std' in v||'n' in v))
      return `<div class="row"><div><strong>${esc(k)}</strong><small>mean ${esc(v.mean)} · std ${esc(v.std)} · n ${esc(v.n)}</small></div><span class="tag">raw</span></div>`;
    return `<div class="row"><div><strong>${esc(k)}</strong><small class="mono">${esc(typeof v==='object'?JSON.stringify(v).slice(0,120):v)}</small></div></div>`;
  }).join('');
}

function renderExperiments(){
  const exps=experimentIndex();
  $('expBoardTag').textContent=`${exps.length} listed`;
  if(!exps.length){
    $('expBoard').innerHTML='<div class="empty">No experiments in the plan or execution ledger yet.</div>';
    $('expDetailGrid').innerHTML='';
    return;
  }
  $('expBoard').innerHTML=exps.map(x=>{
    const path=x.artifact?.raw_results_path||x.output?.raw_results_path||'';
    return `<div class="exp-row">
      <div><div class="name">${esc(x.name)}</div><div class="meta">${esc(x.spec?.type||x.contract?.split_policy||'experiment')}</div></div>
      <div><span class="tag ${x.listed?'good':'idle'}">${x.listed?'yes':'—'}</span></div>
      <div><span class="tag ${x.ran?(x.failed?'bad':'live'):'idle'}">${x.ran?(x.failed?'failed':'yes'):'—'}</span></div>
      <div><span class="tag ${x.done?'good':x.ran?'warn':'idle'}">${x.done?'yes':x.ran?'pending':'—'}</span></div>
      <div>${path?copyRow(path):'<span class="note">No artifact path</span>'}</div>
    </div>`;
  }).join('');

  $('expDetailGrid').innerHTML=exps.map(x=>{
    const seeds=x.artifact?.environment?.seeds||x.contract?.seeds||x.spec?.seeds||[];
    const metrics=x.artifact?.seed_results?.aggregate_metrics||x.output?.aggregate_metrics||x.report?.metrics||{};
    const prediction=x.spec?.falsifiable_prediction||x.contract?.hypothesis||x.spec?.description||'';
    const hash=x.artifact?.content_hash||x.contract?.contract_hash||x.output?.contract_hash||'';
    const path=x.artifact?.raw_results_path||x.output?.raw_results_path||'';
    return `<article class="card span-6">
      <div class="card-head"><h3>${esc(x.name)}</h3><span class="tag ${tagFor(x.status)}">${esc(x.status)}</span></div>
      ${prediction?`<p class="note">${esc(prediction)}</p>`:''}
      <div class="kv"><span class="k">Seeds</span><span class="v mono">${esc(seeds.length?seeds.join(', '):'—')}</span><span></span></div>
      <div class="kv"><span class="k">Hash</span><span class="v mono" title="${esc(hash)}">${esc(hash?short(hash,28):'—')}</span>${hash?copyBtn(hash):'<span></span>'}</div>
      <div class="kv"><span class="k">Results</span><span class="v mono" title="${esc(path)}">${esc(path?short(path,36):'—')}</span>${path?copyBtn(path):'<span></span>'}</div>
      ${metricsBlock(metrics)}
      ${(x.spec?.baselines||x.contract?.baselines||[]).length?`<p class="note">Baselines: ${esc((x.spec?.baselines||x.contract?.baselines||[]).join(', '))}</p>`:''}
    </article>`;
  }).join('');
}

function renderAgents(){
  const c=dashboard.capabilities||{};
  $('agentGrid').innerHTML=Object.entries(c).map(([name,m])=>`<article class="card span-6 agent"><div class="card-head"><h3>${esc(name)}</h3><span class="tag">${esc(m.role)}</span></div><div class="metric-label">Allowed capabilities</div><div class="cap-list">${(m.allowed_capabilities||[]).map(x=>`<span class="cap">${esc(x)}</span>`).join('')}</div><div class="metric-label" style="margin-top:14px">Forbidden capabilities</div><div class="cap-list">${(m.forbidden_capabilities||[]).map(x=>`<span class="cap blocked">${esc(x)}</span>`).join('')}</div></article>`).join('')||'<div class="empty">No manifests available.</div>';
}

function renderPaper(){
  const w=workspace();
  const s=w.draft_sections||w.paper?.sections||{};
  const scores=w.supervisor_scores||{};
  $('paperScores').innerHTML=Object.keys(scores).length?Object.entries(scores).map(([k,v])=>`<div class="chip"><span class="n">${esc(v)}</span><span class="l">${esc(k)}</span></div>`).join(''):'';
  $('paperContent').innerHTML=Object.keys(s).length?Object.entries(s).map(([n,c])=>{
    const body=String(c??'');
    return `<h3>${esc(n)} ${copyBtn(body,'copy')}</h3><div>${esc(body).replaceAll('\n','<br>')}</div>`;
  }).join(''):'<div class="empty">The manuscript will appear after verified evidence is available.</div>';
}

function renderActivity(){
  const ev=(dashboard.events_tail||[]).slice().reverse();
  $('eventStream').innerHTML=ev.length?ev.map(e=>`<div class="event"><time>${esc((e.ts||'').slice(11,19))}</time><strong>${esc(e.agent||'system')}</strong> <span class="tag">${esc(e.type)}</span><br><span class="mono">${esc(JSON.stringify(e.data||{}).slice(0,220))}</span></div>`).join(''):'<div class="empty">No events yet.</div>';

  const entries=scratch.entries||[];
  $('scratchpad').innerHTML=entries.length?entries.slice().reverse().map(x=>`<div class="event"><strong>${esc(x.agent||'?')}/${esc(x.kind||'note')}</strong><br>${esc(typeof x.content==='string'?x.content:JSON.stringify(x.content)).slice(0,400)}</div>`).join(''):'<div class="empty">Scratchpad is empty.</div>';

  const cr=dashboard.cross_run||{};
  const lessons=[...(cr.rejections||[]),...(cr.pivots||[])];
  $('crossRun').innerHTML=lessons.length?lessons.map(x=>`<div class="row"><div><strong>${esc(x.item||x.experiment||'lesson')}</strong><small>${esc(x.reason||'')}</small></div></div>`).join(''):'<div class="empty">No cross-run lessons yet.</div>';

  const debate=(workspace().debates||[]).at(-1);
  const scores=workspace().supervisor_scores||{};
  if(!debate&&!Object.keys(scores).length){
    $('debateSnap').innerHTML='<div class="empty">Debate and supervisor signals appear after those phases run.</div>';
  }else{
    const ensemble=(debate?.ensemble_scores||[]).map(x=>Number(x)).filter(n=>!Number.isNaN(n));
    $('debateSnap').innerHTML=`
      <div class="row"><div><strong>Debate rounds</strong></div><span class="mono">${esc(debate?.rounds?.length??'—')}</span></div>
      <div class="row"><div><strong>Ensemble score</strong></div><span class="mono">${esc(ensemble.length?(ensemble.reduce((a,b)=>a+b,0)/ensemble.length).toFixed(2):'—')}</span></div>
      ${Object.entries(scores).map(([k,v])=>`<div class="row"><div><strong>${esc(k)}</strong><small>supervisor</small></div><span class="mono">${esc(v)}</span></div>`).join('')}`;
  }
}

async function renderRuns(){
  const data=await api('/api/runs');
  $('runTable').innerHTML=`<table><thead><tr><th>Run</th><th>Status</th><th>Phase</th><th>Started</th><th></th></tr></thead><tbody>${
    (data.runs||[]).map(r=>`<tr>
      <td><div class="copy-row"><span class="mono">${esc(r.run_id)}</span>${copyBtn(r.run_id)}</div></td>
      <td><span class="tag ${tagFor(r.status)}">${esc(r.status)}</span></td>
      <td>${esc(r.phase||'')}</td>
      <td class="mono">${esc(r.started_at||'')}</td>
      <td><button data-select-run="${esc(r.run_id)}">Select</button> <button class="primary" data-rerun-run="${esc(r.run_id)}">Re-run</button> <button class="danger" data-delete-run="${esc(r.run_id)}">Delete</button></td>
    </tr>`).join('')
  }</tbody></table>`;
  document.querySelectorAll('[data-select-run]').forEach(button=>button.onclick=async()=>{
    selectedRunId=button.dataset.selectRun;await refresh();showView('overview');
  });
  document.querySelectorAll('[data-rerun-run]').forEach(button=>button.onclick=async()=>{
    const runId=button.dataset.rerunRun;
    if(!confirm(`Re-run from checkpoint for run ${runId}? This will resume from where it left off.`))return;
    try{
      await api(`/api/run/resume/${encodeURIComponent(runId)}`,{method:'POST'});
      selectedRunId=runId;
      await refresh();showView('overview');
    }catch(e){alert(e.message)}
  });
  document.querySelectorAll('[data-delete-run]').forEach(button=>button.onclick=async()=>{
    if(!confirm(`Delete run ${button.dataset.deleteRun}?`))return;
    try{
      await api(`/api/runs/${encodeURIComponent(button.dataset.deleteRun)}`,{method:'DELETE'});
      if(selectedRunId===button.dataset.deleteRun)selectedRunId=null;
      await refresh();await renderRuns();
    }catch(e){$('settingsNote').textContent=e.message}
  });
}

function bindCopies(root=document){
  root.querySelectorAll('[data-copy]').forEach(btn=>{
    if(btn._bound) return;
    btn._bound=true;
    btn.onclick=()=>copyText(btn.getAttribute('data-copy'),btn);
  });
}

function render(){
  renderOverview();
  renderExperiments();
  renderEvidence();
  renderAgents();
  renderPaper();
  renderActivity();
  $('dbPath').textContent=dashboard.config?.research_db_path||window.settingsValues?.research_db_path||'local ledger';
  bindCopies();
}

async function refresh(){
  try{
    const query=selectedRunId?`?run_id=${encodeURIComponent(selectedRunId)}`:'';
    [dashboard,status,scratch]=await Promise.all([
      api(`/api/dashboard${query}`),
      api('/api/run/status'),
      api(`/api/scratchpad?limit=30${selectedRunId?`&run_id=${encodeURIComponent(selectedRunId)}`:''}`),
    ]);
    render();
    if($('runs').classList.contains('active')) await renderRuns();
  }catch(e){$('settingsNote').textContent=e.message}
}

async function saveSettingsForm(){
  const keys={};
  settingDefs.forEach(([id])=>{
    const value=settingValue(id);
    if(value!==''&&value!==undefined) keys[id.toUpperCase()]=value;
  });
  const result=await api('/api/keys',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keys})});
  await loadSettings();
  $('providerStatus').className='tag good';
  $('providerStatus').textContent='Saved OK';
  $('settingsNote').textContent=`Saved ${result.saved?.length??Object.keys(keys).length} settings. Live runtime, keys.json, and .env are updated.`;
  return result;
}

$('refreshButton').onclick=refresh;
$('copyRunId').onclick=()=>copyText(dashboard.run_id,$('copyRunId'));
$('copyExpSummary').onclick=()=>{
  const lines=experimentIndex().map(x=>`${x.name}\t${x.status}\t${x.artifact?.raw_results_path||x.output?.raw_results_path||''}`);
  copyText(lines.join('\n')||'No experiments');
};
$('copyPaperBtn').onclick=()=>{
  const s=workspace().draft_sections||workspace().paper?.sections||{};
  const text=Object.entries(s).map(([n,c])=>`# ${n}\n\n${c}`).join('\n\n');
  copyText(text||'');
};

$('startButton').onclick=async()=>{
  try{
    await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:settingValue('research_domain')||null,provider:settingValue('llm_provider')||undefined})});
    refresh();
  }catch(e){alert(e.message)}
};

$('resetOutputsButton').onclick=async()=>{
  if(!confirm('Clear generated outputs only? History is preserved.'))return;
  const confirmation=prompt('Type RESET_OUTPUTS to confirm:');
  if(confirmation!=='RESET_OUTPUTS')return;
  try{await api('/api/data/reset/outputs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmation})});await refresh()}
  catch(e){$('settingsNote').textContent=e.message}
};

$('resetCatalogButton').onclick=async()=>{
  if(!confirm('This separately deletes catalog assets. Continue?'))return;
  const confirmation=prompt('Type DELETE_DATASET_CATALOG to confirm:');
  if(confirmation!=='DELETE_DATASET_CATALOG')return;
  try{await api('/api/data/reset/catalog',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmation})});await refresh()}
  catch(e){$('settingsNote').textContent=e.message}
};

$('saveSettings').onclick=async()=>{
  try{await saveSettingsForm()}
  catch(e){$('providerStatus').className='tag bad';$('providerStatus').textContent='Save failed';$('settingsNote').textContent=e.message}
};

$('testProvider').onclick=async()=>{
  try{
    $('providerStatus').className='tag warn';$('providerStatus').textContent='Testing...';
    await saveSettingsForm();
    const result=await api('/api/keys/test',{method:'POST'});
    $('providerStatus').className='tag good';$('providerStatus').textContent='OK';
    $('settingsNote').textContent=`LLM connection OK: ${result.response||'OK'} · provider=${result.provider||settingValue('llm_provider')}`;
  }catch(e){
    $('providerStatus').className='tag bad';$('providerStatus').textContent='Failed';
    $('settingsNote').textContent=`LLM connection failed: ${e.message}`;
  }
};

document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>{
  showView(b.dataset.view);
  if(b.dataset.view==='runs') renderRuns();
  if(b.dataset.view==='settings') loadSettings().catch(e=>$('settingsNote').textContent=e.message);
});

// Server-Sent Events for real-time updates
const source = new EventSource('/api/admin/stream');
source.onmessage = function(event) {
  try {
    const data = JSON.parse(event.data);
    if (data.type === 'dashboard') {
      render();
    }
  } catch (e) {
    console.error('Failed to parse SSE data', e);
  }
};
source.onerror = function(err) {
  console.error('SSE connection error', err);
};

(async()=>{
  try{await loadSettings()}catch(e){$('settingsNote').textContent=e.message}
  await refresh();
})();
setInterval(refresh,3000);
