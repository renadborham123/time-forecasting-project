(() => {
  let history=[],busy=false,controller=null,generation=0;
  const $=s=>document.querySelector(s),messages=()=>$('#chat-messages');
  function controls(){const available=!!window.selectedCustomer&&!$('#chat-input').disabled;$('#send-chat').disabled=busy||!available||!$('#chat-input').value.trim();document.querySelectorAll('.suggestion').forEach(b=>b.disabled=busy||!available);$('#draft-message').disabled=busy||!available;$('#chat-form').setAttribute('aria-busy',String(busy));}
  function reveal(){$('#chat-body').hidden=false;$('#chat-panel').classList.add('conversation-started');}
  function scroll(){messages().lastElementChild?.scrollIntoView({block:'nearest',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});}
  function message(text,role){
    reveal();const node=document.createElement('div');node.className=role==='user'?'user-bubble':'response-bubble';
    if(role==='user')node.textContent=text;
    else{const label=document.createElement('span');label.className='message-author';label.textContent='ChurnScope assistant';const content=document.createElement('div');content.className='response-text';content.dir='auto';content.textContent=text;node.append(label,content);}
    messages().append(node);scroll();
  }
  function typing(){const node=document.createElement('div');node.className='response-bubble';node.id='typing';node.setAttribute('aria-label','Preparing a response from learner evidence');node.innerHTML='<div class="typing" aria-hidden="true"><i></i><i></i><i></i></div>';messages().append(node);scroll();}
  async function send(text,quick=false){
    text=text.trim();if(!text||!window.selectedCustomer||busy||$('#chat-input').disabled)return;
    const customer=window.selectedCustomer,version=generation;busy=true;controller=new AbortController();message(text,'user');history.push({role:'user',content:text});typing();controls();$('#agent-status').textContent=quick?'Reading verified learner evidence…':'Local agent is answering…';
    try{
      const data=await window.api('/api/customer/'+customer+'/chat',{method:'POST',signal:controller.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text,conversation:history.slice(0,-1).slice(-8),quick})});
      if(version!==generation||customer!==window.selectedCustomer)return;
      $('#typing')?.remove();message(data.answer,'assistant');history.push({role:'assistant',content:data.answer});$('#agent-status').textContent=data.status;
      if(data.recommendation)window.showRecommendation?.(data.recommendation,customer);
    }catch(error){if(version!==generation)return;$('#typing')?.remove();message(error.name==='AbortError'?'The request was interrupted. Try a suggested question.':'The answer could not be loaded. Please try again.','assistant');$('#agent-status').textContent='Please try again';}
    finally{if(version===generation){busy=false;controller=null;controls();}}
  }
  function reset(customer){history=[];generation++;controller?.abort();controller=null;busy=false;$('#chat-input').value='';$('#chat-input').disabled=true;messages().innerHTML='';$('#chat-body').hidden=true;$('#chat-panel').classList.remove('conversation-started');$('#agent-status').textContent=customer?'Ready to discuss '+customer:'Ready';controls();}
  window.Chat={reset,send,refresh:controls,bind(){
    $('#chat-form').addEventListener('submit',e=>{e.preventDefault();const input=$('#chat-input'),value=input.value;if(value.trim()&&!busy){input.value='';send(value);}});
    $('#chat-input').addEventListener('input',controls);
    $('#chat-input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('#chat-form').requestSubmit();}});
    document.querySelectorAll('.suggestion').forEach(b=>b.onclick=()=>send(b.dataset.question,true));
  }};
})();
