(() => {
  const segmentOrder = ['Power Users','Core Engaged','Fading Engagement','Low Activity','Irregular Users'];
  const segmentColors = ['#0f766e','#2563eb','#7c3aed','#b45309','#be185d'];
  const riskColors = {STABLE:'#16a34a',WATCHLIST:'#d97706','AT RISK':'#ea580c',CRITICAL:'#dc2626'};
  let svg, pointLayer, boundaryLayer, labelLayer, points=[], segmented=false, classified=false, animating=false, riskView=false, rawView=false;
  let segmentFilter=null, riskFilter=null, selected=null, focused=0, onSelect=()=>{}, seed=Math.random();
  const reducedMotion=()=>matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $=s=>document.querySelector(s);
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const color=p=>rawView?'#a1a1aa':classified&&riskView?riskColors[p.risk]:segmented?segmentColors[segmentOrder.indexOf(p.segment)]:'#a1a1aa';
  function random(index,salt=0){const value=Math.sin((index+1)*127.1+seed*8192+salt*311.7)*43758.5453;return value-Math.floor(value);}
  function groups(){return segmentOrder.map(name=>[name,points.filter(p=>p.segment===name)]).filter(g=>g[1].length);}
  function relax(nodes,collide,strength=.08){
    nodes.forEach(p=>{p.vx=0;p.vy=0;});
    const simulation=d3.forceSimulation(nodes).stop().force('x',d3.forceX(p=>p.targetX).strength(strength))
      .force('y',d3.forceY(p=>p.targetY).strength(strength)).force('collide',d3.forceCollide(collide).iterations(2));
    for(let i=0;i<90;i++)simulation.tick();
  }
  function scatter(w,h){
    points.forEach((p,i)=>{p.targetX=28+random(i)*(w-56);p.targetY=24+random(i,1)*(h-48);p.x=p.targetX;p.y=p.targetY;p.r=3.8;});
    relax(points,5,.1);
    points.forEach(p=>{p.x=Math.max(16,Math.min(w-16,p.x));p.y=Math.max(16,Math.min(h-16,p.y));});
  }
  function outline(name,members){
    const cx=d3.mean(members,p=>p.x),cy=d3.mean(members,p=>p.y);
    const hull=d3.polygonHull(members.flatMap(p=>[[p.x-13,p.y],[p.x+13,p.y],[p.x,p.y-13],[p.x,p.y+13]]));
    return {name,count:members.length,cx,cy,labelY:d3.max(members,p=>p.y)+33,hull};
  }
  function groupedPositions(w,h){
    const mobile=w<540;
    const anchors=mobile?[[.26,.13],[.72,.27],[.25,.48],[.73,.63],[.38,.84]]:[[.17,.24],[.49,.25],[.83,.23],[.29,.70],[.73,.70]];
    return groups().map(([name,members],i)=>{
      const cx=w*anchors[i][0],cy=h*anchors[i][1],rx=mobile?w*.14:w*.115,ry=mobile?h*.062:h*.11;
      // Jittered clouds preserve membership without implying scientific distances.
      members.forEach((p,j)=>{
        const index=Number(p.customer_id.slice(1)),angle=random(index,2)*Math.PI*2;
        const radius=Math.sqrt(random(index,3))*(.65+.45*random(index,4));
        p.targetX=cx+Math.cos(angle)*rx*radius;
        p.targetY=cy+Math.sin(angle)*ry*radius*(.8+.3*Math.sin(angle*3+i));
        p.x=p.targetX;p.y=p.targetY;p.r=mobile?3.5:4;
      });
      relax(members,mobile?5.2:6,.07);
      members.forEach(p=>{p.x=Math.max(20,Math.min(w-20,p.x));p.y=Math.max(20,Math.min(h-54,p.y));});
      return outline(name,members);
    });
  }
  function similarityPositions(w,h){
    const sx=d3.scaleLinear().domain(d3.extent(points,p=>+p.embedding_x)).nice().range([32,w-32]);
    const sy=d3.scaleLinear().domain(d3.extent(points,p=>+p.embedding_y)).nice().range([h-48,32]);
    points.forEach(p=>Object.assign(p,{x:sx(+p.embedding_x),y:sy(+p.embedding_y),r:3.8}));
    return groups().map(([name,members])=>({...outline(name,members),labelY:d3.mean(members,p=>p.y)-18}));
  }
  function matches(p){return (!segmentFilter||p.segment===segmentFilter)&&(!riskFilter||p.risk===riskFilter);}
  function applyFilters(){
    if(!pointLayer)return;
    if(!points[focused]||!matches(points[focused]))focused=Math.max(0,points.findIndex(matches));
    pointLayer.selectAll('.customer-point').classed('filtered',p=>!matches(p)).classed('dimmed',p=>selected&&p.customer_id!==selected).classed('selected',p=>p.customer_id===selected)
      .attr('tabindex',(p,i)=>points.length&&matches(p)&&i===focused?0:-1).attr('aria-pressed',p=>String(p.customer_id===selected));
    $('#filter-count').textContent=points.length?points.filter(matches).length+' of '+points.length+' learners':'0 learners';
    $('#clear-filter').hidden=!segmentFilter&&!riskFilter;
    document.querySelectorAll('.segment-row').forEach(b=>{b.classList.toggle('active',b.dataset.segment===segmentFilter);b.setAttribute('aria-pressed',String(b.dataset.segment===segmentFilter));});
    document.querySelectorAll('.risk-row').forEach(b=>{b.classList.toggle('active',b.dataset.risk===riskFilter);b.setAttribute('aria-pressed',String(b.dataset.risk===riskFilter));});
  }
  function nodes(){
    const join=pointLayer.selectAll('g.customer-point').data(points,p=>p.customer_id);join.exit().remove();
    const entering=join.enter().append('g').attr('class','customer-point').attr('role','button').attr('transform',p=>'translate('+p.x+','+p.y+')')
      .on('mouseenter',showTip).on('mousemove',moveTip).on('mouseleave',hideTip)
      .on('focus',function(event,p){showTip(event,p,this);}).on('blur',hideTip)
      .on('click',(event,p)=>{if(segmented&&!animating)select(p.customer_id,true);})
      .on('keydown',function(event,p){
        if(event.key==='Enter'||event.key===' '){event.preventDefault();if(segmented&&!animating)select(p.customer_id,true);}
        else if(['ArrowRight','ArrowLeft','ArrowUp','ArrowDown'].includes(event.key)){
          event.preventDefault();const available=points.filter(matches),index=available.findIndex(v=>v.customer_id===p.customer_id);
          const next=available[(index+(event.key==='ArrowRight'||event.key==='ArrowDown'?1:-1)+available.length)%available.length];
          focused=points.indexOf(next);applyFilters();pointLayer.selectAll('.customer-point').filter(v=>v.customer_id===next.customer_id).node()?.focus();
        }
      });
    entering.append('circle').attr('class','point-hit').attr('r',10);
    entering.append('circle').attr('class','point-ring');
    entering.append('circle').attr('class','point-core').style('fill','#a1a1aa');
    return entering.merge(join).attr('aria-label',p=>p.customer_id+', '+(segmented?p.segment:'not segmented')+(classified?', '+p.risk+' risk':''));
  }
  async function render(duration=0){
    if(!svg)return;
    const w=$('.map-stage').clientWidth,h=$('.map-stage').clientHeight;
    if(!w||!h)return;
    svg.attr('viewBox','0 0 '+w+' '+h);
    const similarity=segmented&&!rawView&&$('#layout-select').value==='similarity';
    let outlines=[];
    if(segmented&&!rawView)outlines=similarity?similarityPositions(w,h):groupedPositions(w,h);else scatter(w,h);
    const all=nodes();
    all.select('.point-core').attr('r',p=>p.r).style('fill',color);
    all.select('.point-hit').attr('r',p=>p.r+1);
    all.select('.point-ring').attr('r',p=>p.r+3).style('pointer-events','none');
    boundaryLayer.selectAll('*').remove();labelLayer.selectAll('*').remove();
    const path=d3.line().curve(d3.curveCatmullRomClosed.alpha(.5));
    outlines.forEach(g=>{
      const stroke=segmentColors[segmentOrder.indexOf(g.name)];
      boundaryLayer.append('path').attr('class','group-boundary').attr('d',path(g.hull)).style('stroke',stroke).style('fill',stroke).style('fill-opacity',similarity?.025:.035);
      labelLayer.append('text').attr('class','cluster-label').attr('x',g.cx).attr('y',g.labelY).text(g.name);
      if(!similarity)labelLayer.append('text').attr('class','cluster-count').attr('x',g.cx).attr('y',g.labelY+18).text(g.count+' learners');
    });
    $('#map-subtitle').textContent=!points.length?'Nothing runs until you start.':!segmented||rawView?'Mixed learners. Positions are randomly scattered.':similarity?'PCA positions show behavioral similarity; groups can overlap.':'Outlined clouds show membership. Their spacing is illustrative.';
    applyFilters();
    const time=reducedMotion()?0:duration;
    if(time){
      boundaryLayer.style('opacity',0).transition().delay(time*.65).duration(time*.35).style('opacity',1);
      labelLayer.style('opacity',0).transition().delay(time*.65).duration(time*.35).style('opacity',1);
      await all.transition('positions').duration(time).ease(d3.easeCubicInOut).attr('transform',p=>'translate('+p.x+','+p.y+')').end();
    }else{all.attr('transform',p=>'translate('+p.x+','+p.y+')');boundaryLayer.style('opacity',1);labelLayer.style('opacity',1);}
  }
  function showTip(event,p,element){
    const tooltip=$('#map-tooltip');
    tooltip.innerHTML='<strong>'+escape(p.customer_id)+'</strong><p>'+(segmented?escape(p.segment):'Waiting for segmentation')+'</p><p>'+(classified?escape(p.risk)+' · '+(p.churn_probability*100).toFixed(1)+'% churn':'Risk not classified yet')+'</p><p>'+Math.round(p.recent_minutes)+' min / week</p>';
    tooltip.style.display='block';
    if(element){const rect=element.getBoundingClientRect();moveTip({clientX:rect.x,clientY:rect.y});}else moveTip(event);
  }
  function moveTip(event){const rect=$('.map-stage').getBoundingClientRect(),tip=$('#map-tooltip');tip.style.left=Math.max(8,Math.min(rect.width-tip.offsetWidth-8,event.clientX-rect.left+14))+'px';tip.style.top=Math.max(8,Math.min(rect.height-tip.offsetHeight-8,event.clientY-rect.top+14))+'px';}
  function hideTip(){$('#map-tooltip').style.display='none';}
  function select(id,notify=false){selected=id;focused=Math.max(0,points.findIndex(p=>p.customer_id===id));hideTip();applyFilters();if(notify)onSelect(points.find(p=>p.customer_id===id));}
  function clearFilters(){segmentFilter=null;riskFilter=null;selected=null;applyFilters();}
  async function update(data,animate=false){
    seed=Math.random();points=data.map(p=>({...p,risk:null,churn_probability:null}));segmented=false;classified=false;riskView=false;selected=null;focused=0;
    $('#layout-select').value='grouped';$('#layout-select').disabled=true;$('#map-empty').hidden=!!points.length;$('#map-count').textContent=points.length?points.length+' learners':'No data loaded';hideTip();clearFilters();await render();
    if(animate&&!reducedMotion()){animating=true;try{await pointLayer.selectAll('.customer-point').style('opacity',0).transition('entry').delay((p,i)=>i*2).duration(260).style('opacity',1).end();}finally{animating=false;await render();}}
  }
  async function segment(status){
    animating=true;
    try{
      const result=await window.api('/api/segment',{method:'POST'});
      const previous=new Map(points.map(p=>[p.customer_id,p]));
      points=result.customers.map(p=>({...p,risk:null,churn_probability:null,x:previous.get(p.customer_id).x,y:previous.get(p.customer_id).y,r:3.8}));
      segmented=true;status('K-Means · assigning group colors','Each dot is receiving its predicted behavior group.');
      const all=nodes();
      await all.select('.point-core').transition('color').delay((p,i)=>reducedMotion()?0:i*4).duration(reducedMotion()?0:220).style('fill',color).end();
      status('K-Means · arranging the result','Moving learners with the same segment color into separate clouds.');
      await render(1200);$('#layout-select').disabled=false;return result;
    }finally{animating=false;await render();}
  }
  async function classify(status){
    animating=true;
    try{
      const result=await window.api('/api/classify',{method:'POST'});
      const positions=new Map(points.map(p=>[p.customer_id,p]));points=result.customers.map(p=>({...p,x:positions.get(p.customer_id).x,y:positions.get(p.customer_id).y,r:positions.get(p.customer_id).r}));
      classified=true;riskView=true;status(result.model+' · visualizing risk','Group positions stay fixed; colors now show the churn risk estimate.');
      await nodes().select('.point-core').transition('color').delay((p,i)=>reducedMotion()?0:i*2.5).duration(reducedMotion()?0:220).style('fill',color).end();return result;
    }finally{animating=false;await render();}
  }
  window.CustomerMap={
    init(callback){svg=d3.select('#customer-map');boundaryLayer=svg.append('g').attr('aria-hidden','true');pointLayer=svg.append('g');labelLayer=svg.append('g').attr('aria-hidden','true');onSelect=callback;new ResizeObserver(()=>{if(!animating)render();}).observe($('.map-stage'));$('#layout-select').onchange=()=>{hideTip();render(500);};$('#clear-filter').onclick=clearFilters;},
    update,render,select,segment,classify,clearFilters,isSegmented:()=>segmented,isClassified:()=>classified,segmentOrder,segmentColors,
    setStage(value){riskView=value===2;rawView=value===0;if(!animating&&pointLayer)pointLayer.selectAll('.point-core').style('fill',color);$('#layout-select').disabled=!segmented||rawView;},
    filterSegment(name){segmentFilter=segmentFilter===name?null:name;riskFilter=null;selected=null;applyFilters();},
    filterRisk(name){riskFilter=riskFilter===name?null:name;segmentFilter=null;selected=null;applyFilters();}
  };
})();
