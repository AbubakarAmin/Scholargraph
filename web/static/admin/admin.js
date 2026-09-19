const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const api=async(path,opts)=>{const r=await fetch(path,opts);const data=await r.json().catch(()=>({}));if(!r.ok)throw Error(typeof data.detail==='string'?data.detail:data.detail?.[0]?.msg||r.statusText);return data};
const short=(s,n=14)=>{const t=String(s??'');return t.length<=n?t:t.slice(0,n)+'…'};
const titleCase=s=>String(s||'').replaceAll('_',' ');

let dashboard={},status={},scratch={},selectedRunId=null,toastTimer=null;
let currentMode='full_research';

const phases=['topic_discovery','hypothesis_debate','planning','data_validation','writing_narrative','engineering','independent_validation','writing_results','supervision','meta_evaluation','editing'];
const qaPhases=['qa_literature_retrieval','qa_answer','qa_verification'];
const titles={overview:'Run overview',qa:'QA Answer',experiments:'Experiments',evidence:'Evidence and lineage',agents:'Agent registry',paper:'Manuscript workspace',activity:'Activity',runs:'Run history',settings:'System settings'};

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

// --- Confirm Dialog ---
function confirmAction(title,message){
  return new Promise(resolve=>{
    $('confirmTitle').textContent=title;
    $('confirmMessage').textContent=message;
    $('confirmOverlay').classList.add('show');
    const cleanup=()=>{$('confirmOverlay').classList.remove('show');$('confirmOk').onclick=null;$('confirmCancel').onclick=null};
    $('confirmOk').onclick=()=>{cleanup();resolve(true)};
    $('confirmCancel').onclick=()=>{cleanup();resolve(false)};
  });
}

// --- Presets ---
const PRESETS_KEY='scholargraph-presets';
function getPresets(){try{return JSON.parse(localStorage.getItem(PRESETS_KEY)||'[]')}catch{return[]}}
function savePresets(presets){localStorage.setItem(PRESETS_KEY,JSON.stringify(presets))}
function renderPresets(){
  const presets=getPresets();
  $('presetList').innerHTML=presets.map(p=>{
    const name=esc(p.name);
    return `<div class="preset-chip" data-preset="${name}">${name}<span class="delete-preset" data-delete-preset="${name}">&times;</span></div>`;
  }).join('');
  document.querySelectorAll('[data-preset]').forEach(el=>{
    el.onclick=async(e)=>{
      if(e.target.dataset.deletePreset){
        const name=e.target.dataset.deletePreset;
        if(await confirmAction('Delete preset',`Delete preset "${name}"?`)){
          savePresets(getPresets().filter(p=>p.name!==name));
          renderPresets();
        }
        return;
      }
      const preset=presets.find(p=>p.name===el.dataset.preset);
      if(preset){
        Object.entries(preset.values||{}).forEach(([k,v])=>{
          const field=$(`setting_${k}`);
          if(field){
            if(field.type==='checkbox')field.checked=v;
            else field.value=v;
          }
        });
        toast(`Loaded preset: ${preset.name}`);
      }
    };
  });
}
$('savePresetBtn').onclick=()=>{
  const name=$('presetNameInput').value.trim();
  if(!name){toast('Enter a preset name');return}
  const keys={};
  settingDefs.forEach(([id])=>{keys[id]=settingValue(id)});
  const presets=getPresets().filter(p=>p.name!==name);
  presets.push({name,values:keys});
  savePresets(presets);
  $('presetNameInput').value='';
  renderPresets();
  toast(`Saved preset: ${name}`);
};

// --- Run Search & Filter ---
let runSearchQuery='';
let runFilterMode='all';
$('runSearchInput').oninput=(e)=>{runSearchQuery=e.target.value.toLowerCase();renderRuns()};
document.querySelectorAll('.run-filter-btn').forEach(btn=>{
  btn.onclick=()=>{
    document.querySelectorAll('.run-filter-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    runFilterMode=btn.dataset.filter;
    renderRuns();
  };
});

// --- Manuscript TOC ---
function renderTOC(){
  const w=workspace();
  const s=w.draft_sections||w.paper?.sections||{};
  const sections=Object.keys(s);
  if(!sections.length){$('tocList').innerHTML='<div class="note">No sections yet</div>';return}
  $('tocList').innerHTML=sections.map((n,i)=>`<a class="toc-item" data-toc-section="${esc(n)}" href="#section-${i}">${esc(n)}</a>`).join('');
  document.querySelectorAll('[data-toc-section]').forEach(el=>{
    el.onclick=(e)=>{
      e.preventDefault();
      document.querySelectorAll('.toc-item').forEach(x=>x.classList.remove('active'));
      el.classList.add('active');
      const target=document.getElementById(`section-${Array.from(document.querySelectorAll('[data-toc-section]')).indexOf(el)}`);
      if(target)target.scrollIntoView({behavior:'smooth',block:'start'});
    };
  });
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

// --- Pagination ---
function paginate(items,page,size){
  const total=Math.ceil(items.length/size);
  const start=page*size;
  return {items:items.slice(start,start+size),total,page,start};
}
function paginatedList(containerId,items,renderFn,opts={}){
  const{pageSize=8,page=0}=opts;
  const p=paginate(items,page,pageSize);
  const el=$(containerId);
  if(!items.length){el.innerHTML='<div class="empty">No entries.</div>';return}
  el.innerHTML=p.items.map(renderFn).join('');
  if(p.total>1){
    el.innerHTML+=`<div class="pagination"><button class="ghost" data-page-prev="${containerId}" ${p.page===0?'disabled':''}>&#8592; Prev</button><span class="note">Page ${p.page+1} of ${p.total}</span><button class="ghost" data-page-next="${containerId}" ${p.page>=p.total-1?'disabled':''}>Next &#8594;</button></div>`;
  }
}
const _pageState={};
function nextPage(containerId,items,renderFn,opts={}){
  const{pageSize=8}=opts;
  const total=Math.ceil(items.length/pageSize);
  _pageState[containerId]=Math.min((_pageState[containerId]||0)+1,total-1);
  paginatedList(containerId,items,renderFn,{pageSize,page:_pageState[containerId]});
}
function prevPage(containerId,items,renderFn,opts={}){
  _pageState[containerId]=Math.max((_pageState[containerId]||0)-1,0);
  paginatedList(containerId,items,renderFn,{pageSize:opts.pageSize||8,page:_pageState[containerId]});
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

function isQAMode(){
  const w=workspace();
  return w.current_phase?.startsWith('qa_') || w.mode === 'qa' || dashboard.phase?.startsWith('qa_');
}

function runMode(r){
  const summary=tryParse(r.summary_json);
  const ws=summary?.workspace||{};
  if(ws.mode==='qa') return 'QA';
  if(ws.current_phase?.startsWith('qa_')) return 'QA';
  if(r.phase?.startsWith('qa_')) return 'QA';
  return 'Research';
}

function renderOverview(){
  const w=workspace();
  const findings=w.verification_findings||[];
  const blocking=findings.filter(x=>x.blocking);
  const exps=experimentIndex();
  const doneCount=exps.filter(x=>x.done).length;
  const listedCount=exps.filter(x=>x.listed||x.ran).length||exps.length;
  const qaMode=isQAMode();
  const qaAnswer=w.qa_answer;
  const qaCitation=w.qa_citation_verification;
  const qaReady=qaAnswer&&qaCitation?.passed;
  const runError=dashboard.release?.error;
  const terminal=dashboard.release?.status==='blocked'||w.terminal_error||w.evidence_gate?.terminal;
  const ready=qaMode?(qaReady||!!qaAnswer):(!blocking.length && w.reproducibility?.passed && Object.keys(w.execution_artifacts||{}).length>0);

  // Onboarding
  const hasRun=dashboard.run_id||w.plan||w.paper||qaAnswer;
  $('onboardingCard').style.display=hasRun?'none':'block';

  const readinessText=runError?'Error':terminal?'Blocked':qaMode?(qaReady?'Answer ready':qaAnswer?'Citations pending':dashboard.phase==='qa_literature_retrieval'?'Retrieving literature':dashboard.phase==='qa_answer'?'Generating answer':'Preparing'):(ready?'Ready':'Blocked');
  $('readiness').textContent=readinessText;
  $('readiness').className='metric-value '+(runError?'bad':terminal?'bad':(qaReady||(ready&&!qaMode))?'good':qaAnswer?'warn':'bad');
  $('readinessNote').textContent=runError?runError:(terminal?(dashboard.release?.reason||w.terminal_error||'Terminal evidence failure'):(qaMode?(qaReady?'Citation-verified synthesis answer':qaAnswer?'Answer generated — some citations unverified':dashboard.phase==='qa_literature_retrieval'?'Searching for relevant papers and sources':'Synthesizing literature-backed answer'):(ready?'Evidence gate passed':'Verification or reproducibility incomplete')));

  $('phaseMetric').textContent=titleCase(dashboard.phase||'idle');
  $('phaseNote').textContent=dashboard.status||'idle';

  const modeLabel=qaMode?'QA Synthesis':w.mode==='qa'?'QA':'Full Research';
  $('modeMetric').textContent=modeLabel;
  $('modeMetric').className='metric-value '+(qaMode?'good':'');
  $('modeNote').textContent=qaMode?'Literature-backed answer':'Full experiment pipeline';

  $('findingMetric').textContent=blocking.length;
  $('findingMetric').className='metric-value '+(blocking.length?'bad':'good');
  $('topicLine').textContent=qaMode?(w.user_query||w.qa_answer?.query||'QA synthesis'):w.plan?.title||w.paper?.topic?.title||'No active research run';

  // Run metrics
  const events=dashboard.events_tail||[];
  const stats=dashboard.tracker_stats||{};
  const llmCalls=stats.llm_calls||0;
  const phaseCount=qaMode?qaPhases:phases;
  const completedPhases=phaseCount.filter((p,i)=>{
    const at=phaseCount.indexOf(dashboard.phase);
    return i<at;
  }).length;
  const tokenEst=llmCalls>0?`~${llmCalls*2000}`:'0';
  $('costTokens').textContent=tokenEst;
  $('costCalls').textContent=llmCalls;
  const startTime=w.started_at?new Date(w.started_at):null;
  if(startTime){
    const mins=Math.floor((Date.now()-startTime.getTime())/60000);
    $('costDuration').textContent=mins>60?`${(mins/60).toFixed(1)}h`:`${mins}m`;
  }else{
    $('costDuration').textContent='0m';
  }
  $('costPhases').textContent=`${completedPhases}/${phaseCount.length}`;

  $('runStatus').textContent=status.running?'Running':status.error?'Error':dashboard.status||'Idle';
  $('runId').textContent=dashboard.run_id?'#'+dashboard.run_id:'';
  $('copyRunId').style.display=dashboard.run_id?'inline-flex':'none';
  $('pulse').className='pulse '+(status.running?'live':status.error?'bad':'');
  $('phaseTag').textContent=titleCase(dashboard.phase||'idle');

  // Pipeline timeline
  const activePhases=qaMode?qaPhases:phases;
  const at=activePhases.indexOf(dashboard.phase);
  let timelineHtml='';
  activePhases.forEach((p,i)=>{
    const isDone=i<at;
    const isActive=p===dashboard.phase;
    const isError=isActive&&status.error;
    timelineHtml+=`<div class="pipeline-node ${isDone?'done':isActive?'active':''}${isError?' error':''}"><div class="pipeline-dot"></div><div class="pipeline-label">${titleCase(p)}</div></div>`;
    if(i<activePhases.length-1){
      timelineHtml+=`<div class="pipeline-connector ${isDone?'done':isActive?'active':''}"></div>`;
    }
  });
  $('pipelineTimeline').innerHTML=timelineHtml;

  // Topic cards
  const topics=w.topics||w.paper?.topic?.alternatives||[];
  const topicList=topics.length?topics:[];
  if(w.paper?.topic)topicList.unshift(w.paper.topic);
  $('topicCount').textContent=`${topicList.length} topics`;
  $('topicCards').innerHTML=topicList.length?topicList.slice(0,6).map(t=>{
    const score=t.score||t.novelty_score||0;
    const scorePercent=Math.round(score*100);
    const scoreColor=scorePercent>70?'var(--green)':scorePercent>40?'var(--yellow)':'var(--red)';
    return `<div class="topic-card">
      <div class="topic-card-title">${esc(t.title||'Untitled')}</div>
      <div class="topic-card-meta">${esc(t.domain||'')} ${t.year?`· ${esc(t.year)}`:''}</div>
      <div class="topic-score-bar"><div class="topic-score-fill" style="width:${scorePercent}%;background:${scoreColor}"></div></div>
      <div class="topic-score-label"><span>Score</span><span>${scorePercent}%</span></div>
      <div class="topic-card-status"><span class="tag ${tagFor(t.status||'discovered')}">${esc(t.status||'discovered')}</span><span class="note">${esc(t.source||'')}</span></div>
    </div>`;
  }).join(''):'<div class="empty">No topics discovered yet. Topics appear after the first research phase.</div>';

  if(qaMode){
    $('gateCard').style.display='none';
    $('expProgressCard').style.display='none';
  }else{
    $('gateCard').style.display='';
    $('expProgressCard').style.display='';
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
  }

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
    ['Run mode',w.mode||'full_research'],
    ['Datasets',Object.keys(w.data_artifacts||{}).length],
    ['Claims', (dashboard.evidence_trace||[]).length],
    ['Supervisor avg',avg],
    ['Debate rounds',debate?.rounds?.length??'—'],
    ['Human approved',w.human_approved?'yes':'no'],
  ].map(([k,v])=>`<div class="row"><div><strong>${esc(k)}</strong></div><span class="mono">${esc(v)}</span></div>`).join('');

  // Show/hide QA nav
  $('navQA').style.display=(w.mode==='qa'||qaMode)?'':'none';
}

function renderQA(){
  const w=workspace();
  const qa=w.qa_answer;
  const lit=w.literature_context;
  const citation=w.qa_citation_verification;

  if(!qa&&!lit){
    $('qaAnswerContent').innerHTML='<div class="empty">No QA answer yet. Start a run in QA mode to generate a synthesis.</div>';
    $('qaCitationDetails').innerHTML='';
    $('qaKeyFindings').innerHTML='';
    $('qaLimitations').textContent='';
    $('qaBibliography').innerHTML='';
    $('qaLiterature').innerHTML='';
    return;
  }

  $('qaQueryLine').textContent=qa?.query||w.user_query||'QA synthesis';

  if(qa){
    $('qaAnswerTag').textContent='complete';
    $('qaAnswerTag').className='tag good';
    const answer=qa.answer||'';
    $('qaAnswerContent').innerHTML=`<div class="qa-answer-text">${esc(answer).replace(/\n\n/g,'</p><p>').replace(/\n/g,'<br>')}</div>`;
  }else{
    $('qaAnswerTag').textContent='pending';
    $('qaAnswerTag').className='tag idle';
    $('qaAnswerContent').innerHTML='<div class="empty">Generating answer...</div>';
  }

  if(citation){
    const passed=citation.passed;
    $('qaCitationTag').textContent=passed?'verified':'issues found';
    $('qaCitationTag').className='tag '+(passed?'good':'warn');
    $('qaCitationDetails').innerHTML=`
      <div class="qa-citation-stat">
        <div class="chip"><span class="n">${citation.score?.toFixed(1)||'—'}</span><span class="l">Score</span></div>
        <div class="chip"><span class="n">${(citation.resolved||[]).length}</span><span class="l">Resolved</span></div>
        <div class="chip"><span class="n">${(citation.failed||[]).length}</span><span class="l">Failed</span></div>
      </div>
      <div class="note">${esc(citation.note||'')}</div>
      ${(citation.failed||[]).length?`<div style="margin-top:10px"><strong style="color:var(--red);font-size:12px">Unresolved citations:</strong>${citation.failed.map(f=>`<div class="mono" style="font-size:11px;margin-top:4px">${esc(f.doi||f.arxiv_id||f.citation||'unknown')}</div>`).join('')}</div>`:''}
      ${(citation.unverifiable||[]).length?`<div style="margin-top:10px"><strong style="color:var(--muted);font-size:12px">Unverifiable (author-year):</strong>${citation.unverifiable.slice(0,5).map(u=>`<div class="mono" style="font-size:11px;margin-top:4px">${esc(u.citation||'')}</div>`).join('')}</div>`:''}
    `;
  }else{
    $('qaCitationTag').textContent='pending';
    $('qaCitationTag').className='tag idle';
    $('qaCitationDetails').innerHTML='<div class="empty">Citation verification pending.</div>';
  }

  const findings=qa.key_findings||[];
  $('qaKeyFindings').innerHTML=findings.length?findings.map(f=>`<div class="qa-finding">${esc(f)}</div>`).join(''):'<div class="empty">No key findings recorded.</div>';
  $('qaLimitations').textContent=qa.limitations||'No limitations noted.';

  const bib=qa.bibliography||[];
  $('qaBibTag').textContent=`${bib.length} sources`;
  _pageState['qaBibliography']=_pageState['qaBibliography']||0;
  paginatedList('qaBibliography',bib,(b,i)=>{
    const id=b.doi||b.arxiv_id||'';
    const link=b.doi?`https://doi.org/${b.doi}`:b.arxiv_id?`https://arxiv.org/abs/${b.arxiv_id}`:'';
    return `<div class="qa-bib-entry">
      <div class="qa-bib-title">${esc(b.title||'Untitled')} ${b.year?`<span class="note">(${esc(b.year)})</span>`:''}</div>
      <div class="qa-bib-meta">${id?`<span class="mono">${esc(id)}</span>${link?` <a href="${esc(link)}" target="_blank" rel="noopener">link</a>`:''}`:'<span class="note">No identifier</span>'}</div>
    </div>`;
  },{pageSize:8,page:_pageState['qaBibliography']});

  const papers=lit?.papers||[];
  $('qaLitTag').textContent=`${papers.length} papers`;
  _pageState['qaLiterature']=_pageState['qaLiterature']||0;
  paginatedList('qaLiterature',papers,(p,i)=>{
    return `<div class="qa-lit-row">
      <div style="flex:1;min-width:0">
        <div class="qa-lit-title">${esc(p.title||'Untitled')} ${p.year?`<span class="note">(${esc(p.year)})</span>`:''} ${p.cited_by_count?`<span class="note">· ${esc(p.cited_by_count)} citations</span>`:''}</div>
        <div class="qa-lit-abstract">${esc((p.abstract||'').slice(0,300))}${(p.abstract||'').length>300?'…':''}</div>
        <div class="mono" style="font-size:11px;margin-top:4px;color:var(--muted)">${p.doi?`doi: ${esc(p.doi)}`:''} ${p.arxiv_id?`arXiv: ${esc(p.arxiv_id)}`:''}</div>
      </div>
    </div>`;
  },{pageSize:8,page:_pageState['qaLiterature']});
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
  const qaMode=isQAMode();
  const qa=w.qa_answer;
  if(qaMode&&qa){
    const bib=qa.bibliography||[];
    const findings=qa.key_findings||[];
    let html=`<div class="paper">`;
    html+=`<div id="section-0" style="scroll-margin-top:80px"><h3>QA Synthesis Answer</h3>`;
    html+=`<p style="color:var(--muted);font-size:13px;margin-bottom:16px">Query: ${esc(qa.query||w.user_query||'')}</p>`;
    html+=`<div>${esc(qa.answer||'').replaceAll('\n\n','</p><p>').replaceAll('\n','<br>')}</div></div>`;
    if(findings.length){
      html+=`<div id="section-1" style="scroll-margin-top:80px;margin-top:24px"><h3>Key Findings</h3><ul>`;
      findings.forEach(f=>{html+=`<li>${esc(f)}</li>`});
      html+=`</ul></div>`;
    }
    if(qa.limitations){
      html+=`<div id="section-2" style="scroll-margin-top:80px;margin-top:24px"><h3>Limitations</h3><p>${esc(qa.limitations)}</p></div>`;
    }
    if(bib.length){
      html+=`<div id="section-${findings.length?2:1}${qa.limitations?1:0}" style="scroll-margin-top:80px;margin-top:24px"><h3>Bibliography</h3>`;
      bib.forEach(b=>{
        const id=b.doi||b.arxiv_id||'';
        const link=b.doi?`https://doi.org/${b.doi}`:b.arxiv_id?`https://arxiv.org/abs/${b.arxiv_id}`:'';
        html+=`<p>${esc(b.title||'Untitled')}${b.year?` (${esc(b.year)})`:''}${id?` <span class="mono" style="font-size:12px;color:var(--muted)">${esc(id)}</span>`:''}${link?` <a href="${esc(link)}" target="_blank" rel="noopener">link</a>`:''}</p>`;
      });
      html+=`</div>`;
    }
    html+=`</div>`;
    $('paperScores').innerHTML='';
    $('paperContent').innerHTML=html;
    renderTOC();
    return;
  }
  const s=w.draft_sections||w.paper?.sections||{};
  const scores=w.supervisor_scores||{};
  $('paperScores').innerHTML=Object.keys(scores).length?Object.entries(scores).map(([k,v])=>`<div class="chip"><span class="n">${esc(v)}</span><span class="l">${esc(k)}</span></div>`).join(''):'';
  $('paperContent').innerHTML=Object.keys(s).length?`<div class="paper">${Object.entries(s).map(([n,c],i)=>{
    const body=String(c??'');
    return `<div id="section-${i}" style="scroll-margin-top:80px"><h3>${esc(n)} ${copyBtn(body,'copy')}</h3><div>${esc(body).replaceAll('\n','<br>')}</div></div>`;
  }).join('')}</div>`:'<div class="paper"><div class="empty"><div class="empty-state-icon">&#128221;</div><div class="empty-state-title">No manuscript yet</div><div class="empty-state-desc">The manuscript will appear here after verified evidence is available and the writer agent has completed drafting.</div></div></div>';
  renderTOC();
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
  let runs=data.runs||[];
  // Search filter
  if(runSearchQuery){
    runs=runs.filter(r=>{
      const mode=runMode(r);
      const searchStr=`${r.run_id} ${mode} ${r.status} ${r.phase||''}`.toLowerCase();
      return searchStr.includes(runSearchQuery);
    });
  }
  // Mode filter
  if(runFilterMode!=='all'){
    runs=runs.filter(r=>{
      const isQA=runMode(r)==='QA';
      return runFilterMode==='qa'?isQA:!isQA;
    });
  }
  $('runTable').innerHTML=`<table><thead><tr><th>Run</th><th>Mode</th><th>Status</th><th>Phase</th><th>Started</th><th></th></tr></thead><tbody>${
    runs.map(r=>{
      const mode=runMode(r);
      return `<tr>
        <td><div class="copy-row"><span class="mono">${esc(r.run_id)}</span>${copyBtn(r.run_id)}</div></td>
        <td><span class="tag">${esc(mode)}</span></td>
        <td><span class="tag ${tagFor(r.status)}">${esc(r.status)}</span></td>
        <td>${esc(r.phase||'')}</td>
        <td class="mono">${esc(r.started_at||'')}</td>
        <td><button data-select-run="${esc(r.run_id)}">Select</button> <button class="primary" data-rerun-run="${esc(r.run_id)}">Re-run</button> <button class="danger" data-delete-run="${esc(r.run_id)}">Delete</button></td>
      </tr>`;
    }).join('')
  }</tbody></table>`+(!runs.length?'<div class="empty">No runs match your search.</div>':'');
  document.querySelectorAll('[data-select-run]').forEach(button=>button.onclick=async()=>{
    selectedRunId=button.dataset.selectRun;await refresh();showView('overview');
  });
  document.querySelectorAll('[data-rerun-run]').forEach(button=>button.onclick=async()=>{
    const runId=button.dataset.rerunRun;
    if(!await confirmAction('Re-run',`Re-run from checkpoint for run ${runId}? This will resume from where it left off.`))return;
    try{
      await api(`/api/run/resume/${encodeURIComponent(runId)}`,{method:'POST'});
      selectedRunId=runId;
      await refresh();showView('overview');
    }catch(e){alert(e.message)}
  });
  document.querySelectorAll('[data-delete-run]').forEach(button=>button.onclick=async()=>{
    if(!await confirmAction('Delete run',`Delete run ${button.dataset.deleteRun}? This cannot be undone.`))return;
    try{
      await api(`/api/runs/${encodeURIComponent(button.dataset.deleteRun)}`,{method:'DELETE'});
      if(selectedRunId===button.dataset.deleteRun)selectedRunId=null;
      await refresh();await renderRuns();
    }catch(e){$('settingsNote').textContent=e.message}
  });
}

function tryParse(s){try{return JSON.parse(s)}catch{return{}}}

function bindCopies(root=document){
  root.querySelectorAll('[data-copy]').forEach(btn=>{
    if(btn._bound) return;
    btn._bound=true;
    btn.onclick=()=>copyText(btn.getAttribute('data-copy'),btn);
  });
}

function render(){
  renderOverview();
  renderQA();
  renderExperiments();
  renderEvidence();
  renderAgents();
  renderPaper();
  renderActivity();
  renderPresets();
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
  }catch(e){$('settingsNote').textContent=e.message}
  render();
  try{if($('runs').classList.contains('active')) await renderRuns()}catch(e){}
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

// --- Mode selector ---
$('runMode').onchange=()=>{
  currentMode=$('runMode').value;
  $('queryGroup').style.display=currentMode==='qa'?'flex':'none';
};

// --- Start run ---
$('startButton').onclick=async()=>{
  try{
    const mode=$('runMode').value;
    const body={domain:settingValue('research_domain')||null,provider:settingValue('llm_provider')||undefined,mode};
    if(mode==='qa'){
      const query=$('runQuery').value.trim();
      if(!query){alert('Please enter a query for QA mode.');return}
      body.query=query;
    }
    await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    refresh();
  }catch(e){alert(e.message)}
};

// --- Dark mode ---
function applyTheme(theme){
  document.documentElement.setAttribute('data-theme',theme);
  localStorage.setItem('scholargraph-theme',theme);
  const btn=$('darkModeToggle');
  if(btn)btn.textContent=theme==='dark'?'light':'dark';
}
function toggleDarkMode(){
  const current=document.documentElement.getAttribute('data-theme');
  applyTheme(current==='dark'?'light':'dark');
}
$('darkModeToggle').onclick=toggleDarkMode;
// Restore saved theme
const savedTheme=localStorage.getItem('scholargraph-theme');
if(savedTheme)applyTheme(savedTheme);

// --- QA export ---
$('copyQaAnswer').onclick=()=>{
  const w=workspace();
  const qa=w.qa_answer;
  if(!qa){toast('No QA answer to copy');return}
  copyText(qa.answer||'');
};
$('exportQaAnswer').onclick=()=>{
  const w=workspace();
  const qa=w.qa_answer;
  if(!qa){toast('No QA answer to export');return}
  const bib=qa.bibliography||[];
  const findings=qa.key_findings||[];
  let md=`# QA Synthesis Answer\n\n`;
  md+=`**Query:** ${qa.query||w.user_query||'—'}\n\n`;
  md+=`## Answer\n\n${qa.answer||''}\n\n`;
  if(findings.length){
    md+=`## Key Findings\n\n`;
    findings.forEach((f,i)=>{md+=`${i+1}. ${f}\n`});
    md+=`\n`;
  }
  if(qa.limitations){md+=`## Limitations\n\n${qa.limitations}\n\n`}
  if(bib.length){
    md+=`## Bibliography\n\n`;
    bib.forEach(b=>{
      md+=`- ${b.title||'Untitled'} (${b.year||'n.d.'})`;
      if(b.doi)md+=` doi:${b.doi}`;
      if(b.arxiv_id)md+=` arXiv:${b.arxiv_id}`;
      md+=`\n`;
    });
    md+=`\n`;
  }
  const w2=w.qa_citation_verification;
  if(w2){
    md+=`## Citation Verification\n\n`;
    md+=`- Score: ${w2.score?.toFixed(1)||'—'}\n`;
    md+=`- Passed: ${w2.passed?'Yes':'No'}\n`;
    md+=`- Note: ${w2.note||'—'}\n`;
  }
  copyText(md);
  toast('Markdown exported to clipboard');
};

// --- Keyboard shortcuts ---
document.addEventListener('keydown',(e)=>{
  // Ignore when typing in inputs
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'||e.target.tagName==='SELECT')return;
  const key=e.key.toLowerCase();
  if(key==='r'&&!e.ctrlKey&&!e.metaKey){e.preventDefault();refresh()}
  if(key==='d'&&!e.ctrlKey&&!e.metaKey){e.preventDefault();toggleDarkMode()}
  if(key==='1'){e.preventDefault();showView('overview')}
  if(key==='2'){e.preventDefault();showView('experiments')}
  if(key==='3'){e.preventDefault();showView('evidence')}
  if(key==='4'){e.preventDefault();showView('paper')}
  if(key==='5'){e.preventDefault();showView('activity')}
  if(key==='6'){e.preventDefault();showView('runs')}
  if(key==='7'){e.preventDefault();showView('settings')}
  if(key==='q'&&!e.ctrlKey&&!e.metaKey){e.preventDefault();showView('qa')}
  if(e.key==='Escape'){document.activeElement?.blur()}
});

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

$('resetOutputsButton').onclick=async()=>{
  if(!await confirmAction('Reset outputs','Clear generated outputs only? History is preserved.'))return;
  const confirmation=prompt('Type RESET_OUTPUTS to confirm:');
  if(confirmation!=='RESET_OUTPUTS')return;
  try{await api('/api/data/reset/outputs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmation})});await refresh()}
  catch(e){$('settingsNote').textContent=e.message}
};

$('resetCatalogButton').onclick=async()=>{
  if(!await confirmAction('Reset catalog','This separately deletes catalog assets. Continue?'))return;
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
let sseRetryMs = 1000;
let sseRetryMax = 30000;
function connectSSE() {
  const source = new EventSource('/api/admin/stream');
  source.onmessage = function(event) {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'dashboard' && data.payload) {
        dashboard = data.payload;
        render();
      }
    } catch (e) {
      console.error('Failed to parse SSE data', e);
    }
  };
  source.onerror = function() {
    source.close();
    setTimeout(function() { connectSSE(); }, sseRetryMs);
    sseRetryMs = Math.min(sseRetryMs * 2, sseRetryMax);
  };
  source.onopen = function() {
    sseRetryMs = 1000;
  };
}
connectSSE();

// --- Pagination event delegation ---
document.addEventListener('click',(e)=>{
  const prev=e.target.closest('[data-page-prev]');
  const next=e.target.closest('[data-page-next]');
  if(!prev&&!next)return;
  const id=prev?.dataset.pagePrev||next?.dataset.pageNext;
  if(!id)return;
  const w=workspace();
  const qa=w.qa_answer;
  const lit=w.literature_context;
  if(id==='qaBibliography'){
    const bib=qa?.bibliography||[];
    const renderFn=(b)=>{
      const id2=b.doi||b.arxiv_id||'';
      const link=b.doi?`https://doi.org/${b.doi}`:b.arxiv_id?`https://arxiv.org/abs/${b.arxiv_id}`:'';
      return `<div class="qa-bib-entry"><div class="qa-bib-title">${esc(b.title||'Untitled')} ${b.year?`<span class="note">(${esc(b.year)})</span>`:''}</div><div class="qa-bib-meta">${id2?`<span class="mono">${esc(id2)}</span>${link?` <a href="${esc(link)}" target="_blank" rel="noopener">link</a>`:''}`:'<span class="note">No identifier</span>'}</div></div>`;
    };
    if(prev)prevPage(id,bib,renderFn,{pageSize:8});
    else nextPage(id,bib,renderFn,{pageSize:8});
  }
  if(id==='qaLiterature'){
    const papers=lit?.papers||[];
    const renderFn=(p)=>{
      return `<div class="qa-lit-row"><div style="flex:1;min-width:0"><div class="qa-lit-title">${esc(p.title||'Untitled')} ${p.year?`<span class="note">(${esc(p.year)})</span>`:''} ${p.cited_by_count?`<span class="note">· ${esc(p.cited_by_count)} citations</span>`:''}</div><div class="qa-lit-abstract">${esc((p.abstract||'').slice(0,300))}${(p.abstract||'').length>300?'…':''}</div><div class="mono" style="font-size:11px;margin-top:4px;color:var(--muted)">${p.doi?`doi: ${esc(p.doi)}`:''} ${p.arxiv_id?`arXiv: ${esc(p.arxiv_id)}`:''}</div></div></div>`;
    };
    if(prev)prevPage(id,papers,renderFn,{pageSize:8});
    else nextPage(id,papers,renderFn,{pageSize:8});
  }
});

(async()=>{
  try{await loadSettings()}catch(e){$('settingsNote').textContent=e.message}
  await refresh();
})();
setInterval(refresh,5000);
