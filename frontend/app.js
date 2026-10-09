(() => {
  const $=s=>document.querySelector(s);
  const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let population=[],metrics=null,step=0,maxStep=0,working=false,loaded=false,learner=null,forecastData=null,selectionVersion=0,currentAction=null,toastTimer;
  const stepNames=['Load data','Segmentation','Classification','Forecasting','Ask & act'];
  const modelName=value=>({exp_smoothing:'Exponential Smoothing',naive:'Last-value baseline',arima:'ARIMA',lag_regression:'Lag Regression'}[value]||value);
  window.selectedCustomer=null;
  window.showToast=text=>{clearTimeout(toastTimer);$('#toast').textContent=text;$('#toast').classList.add('visible');toastTimer=setTimeout(()=>$('#toast').classList.remove('visible'),3000);};
  window.api=async(path,options={})=>{
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),60000);
    const cancel=()=>controller.abort();if(options.signal?.aborted)cancel();options.signal?.addEventListener('abort',cancel,{once:true});
    try{const response=await fetch(path,{...options,signal:controller.signal});if(!response.ok){let detail;try{detail=(await response.json()).detail;}catch{}throw new Error(typeof detail==='string'?detail:'Request failed ('+response.status+'). Please try again.');}return await response.json();}
    finally{clearTimeout(timer);options.signal?.removeEventListener('abort',cancel);}
  };
  function showError(error){$('#error-banner').hidden=false;$('#error-text').textContent=error.name==='AbortError'?'The request took too long. Please try again.':error.message||'Could not reach the application.';}
  function clearError(){$('#error-banner').hidden=true;}
  function activity(title,detail){$('#pipeline-activity').hidden=false;$('#activity-title').textContent=title;$('#activity-detail').textContent=detail;$('.map-stage').classList.toggle('processing',step<3);}
  function nextLabel(){
    if(step===0)return loaded?'View learners':'Load Data';
    if(step===1)return CustomerMap.isSegmented()?(CustomerMap.isClassified()?'Review churn risk':'Classify risk'):'Segment learners';
    if(step===2)return CustomerMap.isClassified()?'Explore forecasts':'Classify risk';
    if(step===3)return learner?'Open assistant':'Run forecast';
    return 'Ask about the forecast';
  }
  function updateStage(){
    const segmented=CustomerMap.isSegmented(),classified=CustomerMap.isClassified();
    CustomerMap.setStage(step);
    const titles=[loaded?'Your data is ready.':'Every forecast starts with a history.',segmented?'Different habits. Clearer groups.':'Find the patterns in the crowd.',classified?'Now we know who needs attention.':'Add a picture of churn risk.',learner?'What comes next for '+learner.customer_id+'?':'Turn past activity into a forecast.','Your data has a story. Let’s talk.'];
    const descriptions=['Load the sample learners. Then follow their data, one step at a time.','Watch each learner take a segment color, then move next to similar learners.','Keep the groups. Use the classifier to reveal each learner’s churn risk.','Read the history, understand the selected method, and look four weeks ahead.','A focused conversation about this learner’s forecast, evidence, and next step.'];
    $('#step-eyebrow').textContent='Step '+(step+1)+' of 5 · '+stepNames[step];
    $('#step-title').textContent=titles[step];$('#step-description').textContent=descriptions[step];
    $('#primary-action').hidden=step===4;$('#primary-action').disabled=working;$('#primary-action').setAttribute('aria-busy',String(working));
    $('#primary-action').innerHTML=escape(nextLabel())+' <span aria-hidden="true">→</span>';
    $('#map-screen').hidden=step>2;$('#evidence-screen').hidden=step!==3;$('#action-screen').hidden=step!==4;
    $('#load-summary').hidden=step!==0;$('#segment-summary').hidden=step!==1;$('#risk-summary').hidden=step!==2;
    $('#map-title').textContent=step===0?'Your dataset':step===2&&classified?'Churn risk by segment':segmented?'Behavior groups':loaded?'Mixed learners':'Your dataset';
    $('#back-btn').hidden=step===0;$('#back-btn').disabled=working;$('#restart-btn').disabled=working;
    $('#execute-action').disabled=working||!currentAction;$('#reject-action').disabled=working||!currentAction;
    $('#customer-select').disabled=working;document.querySelectorAll('[data-customer]').forEach(b=>b.disabled=working);
    document.querySelectorAll('.step').forEach((b,i)=>{b.disabled=working||i>maxStep;b.classList.toggle('active',i===step);b.classList.toggle('complete',i<maxStep&&i!==step);if(i===step)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');b.querySelector('.step-number').textContent=i<maxStep&&i!==step?'✓':String(i+1);});
    Guide.update({step,completed:step===0?loaded:step===1?segmented:step===2?classified:!!learner,label:nextLabel(),working,customer:window.selectedCustomer,action:step===4?()=>Chat.send("Explain this learner's usage forecast",true):primaryAction});
  }
  function moveTo(index,focus=true){
    step=index;clearError();updateStage();if(index<3&&!working)CustomerMap.render();
    if(index===3&&forecastData)setEvidenceTab('forecast');
    if(focus){$('#step-title').focus({preventScroll:true});window.scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});}
  }
  async function goStep(index){if(working||index<0||index>maxStep)return;if(index===3&&!learner){await run(async()=>{moveTo(3);await selectCustomer(window.selectedCustomer||'C003');});}else if(index!==4||learner)moveTo(index);}
  async function run(task){
    if(working)return;working=true;clearError();updateStage();
    try{await task();}catch(error){showError(error);}
    finally{working=false;$('#pipeline-activity').hidden=true;$('.map-stage').classList.remove('processing');updateStage();}
  }
  function renderSegments(rows){
    $('#segment-list').innerHTML=CustomerMap.segmentOrder.map((name,i)=>'<button class="segment-row" data-segment="'+escape(name)+'" aria-pressed="false"><i class="segment-dot" style="background:'+CustomerMap.segmentColors[i]+'" aria-hidden="true"></i><span>'+escape(name)+'</span><b>'+rows.filter(p=>p.segment===name).length+'</b></button>').join('');
    $('#segment-list').querySelectorAll('button').forEach(b=>b.onclick=()=>CustomerMap.filterSegment(b.dataset.segment));
    $('#segment-summary-title').textContent='Five behavior groups.';$('#segment-summary-description').textContent='Same color, similar behavior. Select a group to highlight its learners.';
    $('#map-status').textContent='Segmentation complete. Next: estimate churn risk.';$('#map-help').textContent='Group names describe patterns, not exact boundaries in behavior.';
  }
  function renderRisk(counts){
    const names={STABLE:'Stable',WATCHLIST:'Watchlist','AT RISK':'At risk',CRITICAL:'Critical'};
    $('#risk-list').innerHTML=Object.entries(names).map(([risk,label])=>'<button class="risk-row" data-risk="'+risk+'" aria-pressed="false"><i class="risk-dot '+risk.toLowerCase().replaceAll(' ','-')+'" aria-hidden="true"></i><span>'+label+'</span><b>'+counts[risk]+'</b></button>').join('');
    $('#risk-list').querySelectorAll('button').forEach(b=>b.onclick=()=>CustomerMap.filterRisk(b.dataset.risk));
    $('#risk-summary-description').textContent='Colors now show risk, while the original segment groups stay together. Select a dot to forecast that learner.';
    $('#map-status').textContent='Risk estimates ready. Next: forecast the coming four weeks.';
    $('#map-help').textContent='Use C003 for a clear example of declining usage.';$('#inspect-selected').disabled=!window.selectedCustomer;
  }
  function preview(p){
    if(!p)return;
    if(p.customer_id!==window.selectedCustomer){
      ++selectionVersion;learner=null;forecastData=null;maxStep=CustomerMap.isClassified()?3:2;
      window.selectedCustomer=p.customer_id;clearAction();Chat.reset(p.customer_id);updateStage();
    }
    window.selectedCustomer=p.customer_id;CustomerMap.select(p.customer_id);
    $('#selection-preview').hidden=false;$('#selected-map-customer').textContent=p.customer_id;
    $('#selected-map-description').textContent=p.segment+(CustomerMap.isClassified()?' · '+p.risk:'. Classification comes next.');
    $('#inspect-selected').disabled=!CustomerMap.isClassified();
  }
  function clearAction(){currentAction=null;$('#recommendation-inline').hidden=true;$('#action-content').innerHTML='';$('#execute-action').disabled=true;$('#reject-action').disabled=true;$('#action-toast').textContent='';}
  function renderShap(data){const max=Math.max(...data.values.map(v=>Math.abs(v.value)),.0001);$('#shap-list').innerHTML=data.values.map(v=>'<div class="shap-row"><span class="shap-name" title="'+escape(v.feature)+'">'+escape(v.label)+'</span><div class="shap-track"><i class="shap-mid"></i><i class="shap-bar '+(v.value<0?'negative':'')+'" style="width:'+Math.max(1,Math.abs(v.value)/max*48)+'%;'+(v.value<0?'right':'left')+':50%"></i></div><span class="shap-value">'+(v.value>0?'+':'')+v.value.toFixed(3)+'</span></div>').join('');}
  function renderProfile(p){const prefs=p.preferences;const rows=[['Favorite topic',prefs.favorite_topic],['Content type',prefs.preferred_content_type],['Preferred feature',prefs.preferred_feature],['Study time',prefs.preferred_study_time],['Subscription age',p.features.subscription_age_weeks+' weeks'],['Historical engagement',prefs.historical_engagement_level]];$('#profile-grid').innerHTML=rows.map(([label,value])=>'<div class="profile-item"><dt>'+escape(label)+'</dt><dd>'+escape(value)+'</dd></div>').join('');}
  function setEvidenceTab(name,focus=false){document.querySelectorAll('[data-tab]').forEach(b=>{const active=b.dataset.tab===name;b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;$('#panel-'+b.dataset.tab).hidden=!active;if(active&&focus)b.focus();});if(name==='forecast'&&forecastData)ForecastChart.draw(forecastData);}
  async function selectCustomer(id){
    const version=++selectionVersion;window.selectedCustomer=id;learner=null;forecastData=null;maxStep=3;clearAction();Chat.reset(id);CustomerMap.select(id);
    $('#customer-select').value=id;$('#evidence-content').hidden=true;$('#evidence-empty').hidden=false;
    $('#evidence-empty').innerHTML='<h2>Reading '+escape(id)+'’s activity…</h2><p>Fitting '+escape(modelName(metrics.forecasting.selected_model))+' to the observed history.</p>';
    activity(modelName(metrics.forecasting.selected_model)+' · fitting recent activity','Reading the 32-week history for '+id+'. Fitting the latest 16 weeks to forecast the next four.');
    document.querySelectorAll('[data-customer]').forEach(b=>b.classList.toggle('active',b.dataset.customer===id));updateStage();
    const [p,f,s]=await Promise.all([api('/api/customer/'+id),api('/api/customer/'+id+'/forecast'),api('/api/customer/'+id+'/explanation')]);
    if(version!==selectionVersion)return;
    learner=p;forecastData=f;maxStep=4;$('#evidence-empty').hidden=true;$('#evidence-content').hidden=false;
    $('#customer-title').textContent='Learner '+id;$('#segment-value').textContent=p.segment;
    $('#churn-value').textContent=(p.churn_probability*100).toFixed(1)+'%';$('#risk-label').textContent=p.risk;$('#risk-label').dataset.risk=p.risk;
    $('#current-usage').textContent=Math.round(f.current_weekly_usage_minutes)+' min';$('#usage-change').textContent=(f.expected_change_pct>0?'+':'')+f.expected_change_pct.toFixed(0)+'%';
    $('#forecast-summary').textContent=Math.round(f.current_weekly_usage_minutes)+' min / week recently → '+Math.round(f.week_4_minutes)+' min by week '+f.forecast[3].week;
    $('#forecast-model').textContent=modelName(f.model_used)+(f.fallback?' · fallback':' · selected');
    $('#method-story').textContent='Naive, Exponential Smoothing, ARIMA, and Lag Regression were compared on earlier windows. '+modelName(f.selected_global_model)+' won, and is fitted to the latest 16 weeks.';
    $('#forecast-weeks').innerHTML=f.forecast.map(v=>'<div><span>Week '+v.week+'</span><strong>'+Math.round(v.usage_minutes)+' <small>min</small></strong></div>').join('');
    $('#forecast-reading').textContent=f.expected_change_pct<-10?'The forecast points to less activity. This gives us a reason to review the learner’s needs before choosing an action.':f.expected_change_pct>10?'The forecast points to more activity. Check the uncertainty range before interpreting this as a lasting recovery.':'The forecast is fairly steady. Continue monitoring their activity and consider the uncertainty range.';
    $('#context-customer').textContent='Learner '+id;$('#context-details').textContent=p.segment+' · '+p.risk+' · forecast '+(f.expected_change_pct>0?'+':'')+f.expected_change_pct.toFixed(0)+'%';
    $('#chat-input').disabled=false;$('#chat-input').placeholder='Ask about '+id+'…';Chat.refresh();renderShap(s);renderProfile(p);
    activity(modelName(f.model_used)+' · drawing the forecast','Model fit complete. Showing weeks 33–36 and the empirical error range.');
    setEvidenceTab('forecast');updateStage();
    await ForecastChart.finished();
  }
  window.showRecommendation=(result,id)=>{
    if(id!==window.selectedCustomer||learner?.customer_id!==id)return;
    currentAction={customer:id,code:result.action};$('#recommendation-inline').hidden=false;
    $('#action-provider').textContent=result.provider==='deterministic'?'Verified eligibility policy':'Local agent · '+result.provider;
    const evidence=(result.verified_evidence||[]).map(line=>'<li>'+escape(line)+'</li>').join('');
    $('#action-content').innerHTML='<h3 class="action-title">'+escape(result.action_label)+'</h3><p class="action-reason">'+escape(result.reason)+'</p>'+(result.message?'<div class="action-message"><span class="action-message-label">Message draft</span>'+escape(result.message)+'</div>':'<p class="action-message muted">No outreach is recommended.</p>')+(evidence?'<details class="action-evidence"><summary>View supporting evidence</summary><ul>'+evidence+'</ul></details>':'');
    $('#execute-action').disabled=false;$('#reject-action').disabled=false;$('#action-toast').textContent='';
    $('#recommendation-inline').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'nearest'});
  };
  async function executeAction(){if(!currentAction||working)return;const action={...currentAction};await run(async()=>{const result=await api('/api/customer/'+action.customer+'/execute-action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:action.code})});if(action.customer!==window.selectedCustomer||currentAction?.code!==action.code)return;currentAction=null;$('#execute-action').disabled=true;$('#reject-action').disabled=true;$('#action-toast').textContent=result.record.label+' · recorded in this simulation.';showToast('Simulated action recorded.');});}
  function renderMetrics(m){
    const c=m.classification,cl=m.clustering,f=m.forecasting,chosen=c.models[c.selected_model];
    const row=(label,value)=>'<div class="eval-row"><span>'+escape(label)+'</span><b>'+escape(value)+'</b></div>';
    $('#metrics-grid').innerHTML='<section class="metrics-section"><h3>Behavior groups · '+escape(cl.selected_model)+'</h3>'+row('Groups',cl.kmeans_clusters)+row('Silhouette score',cl.kmeans_silhouette.toFixed(3))+'<p class="footnote">Overlapping profiles. Map clouds illustrate membership, not similarity distance.</p></section><section class="metrics-section"><h3>Churn classifier · '+escape(c.selected_model)+'</h3>'+row('Customer holdout',c.holdout_size)+row('Precision / Recall',chosen.precision.toFixed(3)+' / '+chosen.recall.toFixed(3))+row('F1 / ROC-AUC',chosen.f1.toFixed(3)+' / '+chosen.roc_auc.toFixed(3))+row('Cutoff → target window','Week 28 → weeks 29–32')+'</section><section class="metrics-section"><h3>Four-week usage forecast</h3><table class="metrics-table"><thead><tr><th scope="col">Model</th><th scope="col">MAE</th><th scope="col">RMSE</th><th scope="col">sMAPE</th></tr></thead><tbody>'+Object.entries(f.metrics).map(([name,v])=>'<tr class="'+(name===f.selected_model?'selected':'')+'"><th scope="row">'+escape(modelName(name))+(name===f.selected_model?' · selected':'')+'</th><td>'+v.mae.toFixed(1)+'</td><td>'+v.rmse.toFixed(1)+'</td><td>'+v.smape.toFixed(1)+'%</td></tr>').join('')+'</tbody></table><p class="footnote">Lowest walk-forward MAE selects the method. Earlier cutoffs: 20, 24, 28. MAE is average absolute error in minutes.</p></section>';
  }
  async function loadData(){
    if(typeof d3==='undefined'||typeof Chart==='undefined')throw new Error('Chart libraries could not load. Check your connection and reload.');
    activity('Load Data · reading the sample','Loading learner records and the saved model evaluation.');
    const [rows,m]=await Promise.all([api('/api/customers'),api('/api/model-metrics')]);population=rows;metrics=m;
    $('#customer-select').innerHTML='<option value="" disabled>Choose a learner</option>'+rows.map(p=>'<option value="'+escape(p.customer_id)+'">'+escape(p.customer_id)+'</option>').join('');
    renderMetrics(m);$('#model-info').disabled=false;activity('Load Data · drawing the learners','300 learner records loaded. Placing their dots in a mixed, random view.');
    await CustomerMap.update(rows,true);loaded=true;maxStep=1;$('#map-status').textContent='Data loaded. Next: discover behavior groups.';moveTo(1,false);
  }
  async function primaryAction(){
    if(working)return;
    await run(async()=>{
      if(step===0){if(loaded)moveTo(1);else await loadData();}
      else if(step===1&&!CustomerMap.isSegmented()){activity('K-Means · predicting segments','Scaling observed behavior and assigning each learner to a behavior group.');const result=await CustomerMap.segment(activity);maxStep=2;renderSegments(result.customers);}
      else if(step<=2&&!CustomerMap.isClassified()){moveTo(2,false);activity(metrics.classification.selected_model+' · estimating churn','Running the saved classifier on behavior observed through week 28.');const result=await CustomerMap.classify(activity);maxStep=3;CustomerMap.clearFilters();renderRisk(result.counts);}
      else if(step===1)moveTo(2);
      else if(step===2){moveTo(3);await selectCustomer(window.selectedCustomer||'C003');}
      else if(step===3){if(learner)moveTo(4);else await selectCustomer(window.selectedCustomer||'C003');}
    });
  }
  async function reset(){
    ++selectionVersion;population=[];loaded=false;step=0;maxStep=0;learner=null;forecastData=null;window.selectedCustomer=null;clearAction();Chat.reset('');clearError();await CustomerMap.update([]);
    $('#segment-list').innerHTML='';$('#risk-list').innerHTML='';$('#segment-summary-title').textContent='Find the patterns.';$('#segment-summary-description').textContent='First the dots take their group’s color. Then similar learners move together.';
    $('#selection-preview').hidden=true;$('#map-help').textContent='Follow the black button to take the next step.';$('#map-status').textContent='Ready when you are.';$('#customer-select').value='';$('#evidence-content').hidden=true;$('#model-info').disabled=true;Guide.reset();moveTo(0);
  }
  Chat.bind();Guide.bind();CustomerMap.init(preview);CustomerMap.update([]);
  $('#primary-action').onclick=primaryAction;document.querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>goStep(Number(b.dataset.step)));
  $('#back-btn').onclick=()=>goStep(step-1);$('#restart-btn').onclick=()=>{if(!working)reset();};
  $('#customer-select').onchange=e=>{if(e.target.value)run(()=>selectCustomer(e.target.value));};document.querySelectorAll('[data-customer]').forEach(b=>b.onclick=()=>run(()=>selectCustomer(b.dataset.customer)));
  $('#inspect-selected').onclick=()=>{if(!working&&CustomerMap.isClassified())run(async()=>{moveTo(3);await selectCustomer(window.selectedCustomer);});};
  $('#review-evidence').onclick=()=>goStep(3);
  document.querySelectorAll('[data-tab]').forEach(b=>{b.onclick=()=>setEvidenceTab(b.dataset.tab);b.onkeydown=e=>{const tabs=[...document.querySelectorAll('[data-tab]')],i=tabs.indexOf(b);let index;if(e.key==='ArrowRight')index=(i+1)%tabs.length;if(e.key==='ArrowLeft')index=(i+tabs.length-1)%tabs.length;if(e.key==='Home')index=0;if(e.key==='End')index=tabs.length-1;if(index!==undefined){e.preventDefault();setEvidenceTab(tabs[index].dataset.tab,true);}};});
  $('#execute-action').onclick=executeAction;$('#draft-message').onclick=()=>Chat.send('Write a personalized message',true);$('#reject-action').onclick=()=>{clearAction();showToast('Recommendation dismissed.');};
  $('#model-info').onclick=()=>$('#metrics-dialog').showModal();$('#close-metrics').onclick=()=>$('#metrics-dialog').close();
  $('#retry-load').onclick=()=>{if(!loaded)primaryAction();else if(step===3&&!learner)run(()=>selectCustomer(window.selectedCustomer||'C003'));else clearError();};
  document.querySelector('.brand').onclick=e=>{e.preventDefault();goStep(0);};updateStage();
})();
