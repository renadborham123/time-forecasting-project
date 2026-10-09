(() => {
  const $=s=>document.querySelector(s);
  const introductory=[
    'Start by loading the sample data. Each dot represents one learner with a history of activity. Then we’ll look for their patterns.',
    'The learners are mixed together. K-Means assigns a color to each behavior group, then moves dots with the same color next to one another. Try Segment learners.',
    'We have behavior groups. The classifier estimates churn risk. The dots stay in their groups while their colors change to show risk.',
    'The forecast reads one learner’s weekly activity, then projects the next four weeks. We’ll show the history, the selected method, and the uncertainty together.',
    'Choose a suggested question for an instant explanation, or type your own. Review an action inside the conversation before simulating it.'
  ];
  const completed=[
    'The sample data is ready. Continue to see the mixed learners before we group them.',
    'The five colors now sit in separate clouds. These outlines show membership; spacing is only for readability. Next, classify churn risk for the same learners.',
    'Risk colors are visible inside the behavior groups. Choose a dot to use that learner, or explore C003’s declining forecast next.',
    'The solid line is observed activity. The dashed line is the next four weeks; the shaded range shows uncertainty. Open the assistant to discuss this forecast.',
    introductory[4]
  ];
  const agents=[
    {id:'data',name:'Data agent',roles:['Meet your learners','Discover their patterns'],mouth:'M31 45 Q40 52 49 45'},
    {id:'risk',name:'Risk agent',roles:['Understand churn risk'],mouth:'M33 45 Q40 50 47 45'},
    {id:'forecast',name:'Forecast agent',roles:['Look four weeks ahead','Let’s talk about the result'],mouth:'M30 44 Q40 57 50 44'}
  ];
  const reduced=()=>matchMedia('(prefers-reduced-motion: reduce)').matches;
  let state=null,open=false,present=false,controller=null,generation=0,key='',raf=null,entranceTimer,revealTimer,swapTimer,identity=null;
  const agentFor=step=>agents[step<2?0:step===2?1:2];
  function setIdentity(step){
    const agent=agentFor(step);identity=agent.id;
    $('#guide').dataset.agent=agent.id;$('#guide-name').textContent=agent.name;$('#guide-character-name').textContent=agent.name;
    $('#guide-role').textContent=agent.roles[step===1||step===4?1:0];
    $('.agent-mouth path').setAttribute('d',agent.mouth);
    $('#guide-close').setAttribute('aria-label','Close '+agent.name);
    $('#guide-toggle').setAttribute('aria-label',(open?'Close ':'Open ')+agent.name);
    $('#guide').setAttribute('aria-label',agent.name+' · pipeline helper');
  }
  function position(){
    if(!state)return;
    const target=state.step<3?$('.map-inspector'):state.step===3?$('#forecast-anchor'):$('#chat-form');
    if(!target||!target.getClientRects().length)return;
    const rect=target.getBoundingClientRect(),guide=$('#guide'),width=guide.offsetWidth,text=$('#guide-text');
    let height=guide.offsetHeight;
    if(innerWidth<=760&&innerHeight<=820&&state.step<4){
      const available=innerHeight-Math.max(12,$('.stage-heading').getBoundingClientRect().bottom+16)-16;
      const fixedHeight=height-text.getBoundingClientRect().height;
      text.style.maxHeight=Math.max(48,Math.min(92,available-fixedHeight))+'px';
      height=guide.offsetHeight;
    }else text.style.removeProperty('max-height');
    if(state.step===3){
      if(innerWidth>760)target.style.minHeight=Math.max(365,height+40)+'px';
      else target.style.removeProperty('min-height');
    }
    let x,y;
    if(innerWidth<=760){
      x=innerWidth-width-16;
      y=state.step===4?rect.top-height-16:Math.max(rect.top-height+100,$('.stage-heading').getBoundingClientRect().bottom+16);
    }
    else if(state.step<3){x=rect.left;y=rect.bottom-height-8;}
    else {
      x=Math.min(innerWidth-width-24,rect.right+20);y=rect.top+20;
      if(x<rect.right+12){x=rect.right-width;if(state.step===4)y=rect.top-height-16;}
    }
    guide.style.left=Math.max(12,Math.min(innerWidth-width-12,x))+'px';
    guide.style.top=Math.max(12,Math.min(innerHeight-height-16,y))+'px';
    guide.dataset.stage=String(state.step);
  }
  function queuePosition(){if(raf)cancelAnimationFrame(raf);raf=requestAnimationFrame(()=>{raf=null;position();});}
  function setOpen(value){
    open=value;const panel=$('#guide-popup');
    panel.setAttribute('aria-hidden',String(!open));panel.inert=!open;
    $('#guide-toggle').setAttribute('aria-expanded',String(open));
    $('#guide-toggle').setAttribute('aria-label',(open?'Close ':'Open ')+agentFor(state?.step||0).name);
    $('#guide').classList.toggle('is-open',open);
    if(open){loadExplanation();queuePosition();}
  }
  function appear(){
    if(!state)return;
    $('#guide').hidden=false;position();
    requestAnimationFrame(()=>{
      present=true;$('#guide').classList.add('is-present');
      clearTimeout(revealTimer);revealTimer=setTimeout(()=>setOpen(true),reduced()?0:220);
    });
  }
  function enterStage(step,first=false){
    clearTimeout(entranceTimer);clearTimeout(revealTimer);clearTimeout(swapTimer);
    setOpen(false);
    const changedIdentity=identity!==agentFor(step).id;
    if(changedIdentity&&present){
      $('#guide').classList.add('is-switching');
      swapTimer=setTimeout(()=>{setIdentity(step);$('#guide').classList.remove('is-switching');},reduced()?0:160);
    }else setIdentity(step);
    if(first||!present){entranceTimer=setTimeout(appear,reduced()?0:420);}
    else {queuePosition();revealTimer=setTimeout(()=>setOpen(true),reduced()?0:480);}
  }
  async function loadExplanation(){
    if(!state||!open||state.working)return;
    const nextKey=[state.step,state.completed,state.customer||''].join(':');if(nextKey===key)return;
    key=nextKey;const version=++generation;controller?.abort();controller=new AbortController();
    try{
      const data=await window.api('/api/pipeline-guide',{method:'POST',signal:controller.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify({stage:state.step,completed:state.completed,customer_id:state.customer||null})});
      if(version!==generation||nextKey!==key)return;
      $('#guide-text').textContent=data.explanation;$('#guide-provider').textContent=data.fallback?'Verified workflow explanation':'Local LLM · '+data.provider;queuePosition();
    }catch(error){if(version===generation&&error.name!=='AbortError')$('#guide-provider').textContent='Verified workflow explanation';}
  }
  window.Guide={
    update(next){
      const stageChanged=!state||next.step!==state.step,first=!state;
      const contextChanged=stageChanged||next.completed!==state.completed||next.customer!==state.customer;
      state=next;
      const arrow=document.createElement('span');arrow.className='guide-action-arrow';arrow.textContent='→';arrow.setAttribute('aria-hidden','true');
      $('#guide-action').replaceChildren(document.createTextNode(next.label),arrow);$('#guide-action').disabled=next.working;$('#guide').classList.toggle('is-working',next.working);
      if(contextChanged){++generation;controller?.abort();key='';$('#guide-text').textContent=(next.completed?completed:introductory)[next.step];$('#guide-provider').textContent='Verified workflow explanation';}
      if(stageChanged)enterStage(next.step,first);
      queuePosition();loadExplanation();
    },
    reset(){++generation;controller?.abort();key='';state=null;present=false;identity=null;clearTimeout(entranceTimer);clearTimeout(revealTimer);clearTimeout(swapTimer);setOpen(false);$('#guide').classList.remove('is-present','is-switching');$('#guide').hidden=true;},
    bind(){
      $('#guide-close').onclick=()=>{clearTimeout(revealTimer);setOpen(false);$('#guide-toggle').focus();};
      $('#guide-toggle').onclick=()=>{clearTimeout(revealTimer);setOpen(!open);};
      $('#guide-action').onclick=()=>{if(!state?.working)state?.action();};
      $('#guide').onkeydown=e=>{if(e.key==='Escape'){clearTimeout(revealTimer);setOpen(false);$('#guide-toggle').focus();}};
      window.addEventListener('resize',queuePosition);window.addEventListener('scroll',queuePosition,{passive:true});
      new ResizeObserver(queuePosition).observe($('#guide'));
    }
  };
})();
