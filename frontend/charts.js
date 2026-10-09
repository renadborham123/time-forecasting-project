(() => {
  let chart, completion=Promise.resolve(), resolveCompletion=()=>{};
  function draw(data){
    const canvas=document.querySelector('#forecast-chart');if(!canvas||canvas.closest('[hidden]'))return;
    resolveCompletion();if(chart)chart.destroy();
    completion=new Promise(resolve=>{resolveCompletion=resolve;});
    const history=data.history,forecast=data.forecast,n=history.length;
    const labels=[...history.map(x=>`W${x.week}`),...forecast.map(x=>`W${x.week}`)];
    const prefix=[...Array(n-1).fill(null),history[n-1].usage_minutes];
    const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
    const boundary={id:'forecastBoundary',afterDraw(c){const x=c.scales.x.getPixelForValue(n-1),{top,bottom}=c.chartArea,ctx=c.ctx;ctx.save();ctx.setLineDash([4,4]);ctx.strokeStyle='#a1a1aa';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#61616b';ctx.font='12px Geist, Arial';ctx.fillText('Forecast →',x+8,top+12);ctx.restore();}};
    chart=new Chart(canvas,{type:'line',plugins:[boundary],data:{labels,datasets:[
      {label:'Error band',data:[...prefix,...forecast.map(x=>x.upper)],borderWidth:0,pointRadius:0,fill:{target:1,above:'rgba(113,113,122,.12)'},backgroundColor:'rgba(113,113,122,.12)',order:3},
      {label:'Lower range',data:[...prefix,...forecast.map(x=>x.lower)],borderWidth:0,pointRadius:0,fill:false,order:3},
      {label:'Observed usage',data:[...history.map(x=>x.usage_minutes),...Array(forecast.length).fill(null)],borderColor:'#18181b',borderWidth:2,pointRadius:0,pointHoverRadius:4,tension:.2,order:1},
      {label:'Forecast usage',data:[...prefix,...forecast.map(x=>x.usage_minutes)],borderColor:'#71717a',borderWidth:2,borderDash:[6,4],pointRadius:3,pointBackgroundColor:'#71717a',pointHoverRadius:5,tension:.2,order:0}
    ]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:reduced?0:1000,onComplete:()=>resolveCompletion()},interaction:{intersect:false,mode:'index'},plugins:{legend:{display:false},tooltip:{backgroundColor:'#18181b',titleColor:'#fff',bodyColor:'#fff',displayColors:false,padding:12,titleFont:{size:14},bodyFont:{size:12},filter:item=>item.datasetIndex>1,callbacks:{label:c=>`${c.dataset.label}: ${Math.round(c.parsed.y)} minutes`}}},scales:{x:{grid:{display:false},ticks:{color:'#61616b',font:{family:'Geist',size:12},maxTicksLimit:10},border:{color:'#e4e4e7'}},y:{beginAtZero:true,grid:{color:'#e4e4e7'},ticks:{color:'#61616b',font:{family:'Geist',size:12},maxTicksLimit:5},border:{display:false},title:{display:true,text:'Minutes / week',color:'#61616b',font:{family:'Geist',size:12}}}}}});
  }
  window.ForecastChart={draw,finished:()=>matchMedia('(prefers-reduced-motion: reduce)').matches?Promise.resolve():completion};
})();
