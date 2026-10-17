const $=id=>document.getElementById(id);
const arms=['pixel','gaussian','sparse'];
const colors={pixel:'#b85b40',gaussian:'#397aac',sparse:'#2b896e'};
const armNames={pixel:'Pixel',gaussian:'Gaussian JEPA',sparse:'Sparse LpWM'};
const nice=s=>String(s).replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const format=x=>typeof x==='number'?(Math.abs(x)>0&&Math.abs(x)<.001?x.toExponential(2):Number(x.toFixed(4)).toString()):x===null?'—':String(x);
const physical=['block_position_rmse','pusher_position_rmse','angle_mae_rad'];
const defs=[
 {id:'layers',chapter:'Layers',title:'Physical information across the encoder',x:'layer',metrics:[...physical,'participation_ratio','effective_rank','exact_zero_fraction','covariance_trace','leading_variance_fraction','mean_abs_feature_correlation','mean_excess_kurtosis','positive_fraction','constant_features'],filters:{},description:'Frozen linear probes at all 12 blocks, final normalization, projector and linked output. Each line follows one trained full-depth model.',caveat:'512 / 128 / 124 probe episodes; 16 frames each. Covariance geometry measures feature structure, not physical understanding. A low error at a tap does not establish causal use of that information.'},
 {id:'full-depth-sensitivity',chapter:'Layers',title:'Physical and appearance sensitivity',x:'layer',metrics:['ratio','background','physical','background_image_mse','physical_image_mse'],filters:{},description:'Response to a fixed background-color change versus physical state perturbations, normalized by development covariance trace.',caveat:'32 development / 128 test resets. Background and physical perturbations are not image-energy matched. A smaller ratio alone is not a proof of nuisance removal.'},
 {id:'full-depth-candidate_ranking',chapter:'Dynamics',title:'Can frozen features support action ranking?',x:'layer',metrics:['spearman','top1_regret','topk_overlap','best_in_topk','D','parameters'],filters:{predictor:'bilinear',scorer:'predicted_native'},description:'45 small predictor fits: three objectives × five taps × three predictor families. Choose the scoring rule to separate native geometry from physical readout.',caveat:'50 paired candidate banks. True-future scoring is a privileged diagnostic. These newly fitted affine/bilinear/MLP predictors are not the original AdaLN model. D is normalized regret reduction, not success rate.'},
 {id:'full-depth-physical_horizons',chapter:'Dynamics',title:'Physical forecast error over time',x:'horizon',metrics:[...physical,'episodes'],filters:{layer:'linked',predictor:'bilinear',readout:'predicted'},description:'Per-horizon errors from the frozen-feature predictor study. Compare taps, predictor capacity and available readout conditions.',caveat:'Cohorts differ across horizons; H32 has only four episodes. Do not interpret the curve as a fixed-cohort monotonic error law.'},
 {id:'independent-depth-depth_summary',chapter:'Depth',title:'Shallower encoders, trained independently',x:'depth',metrics:['final_success','ever_success','spearman','regret','block_rmse','pusher_rmse','angle_mae','mlp_block_rmse','mlp_pusher_rmse','mlp_angle_mae','encoder_parameters','train_hours'],filters:{},description:'Gaussian and Sparse trained from scratch at depths 1, 2 and 4. Depth 12 is a historical reference. No Pixel depth sweep was run.',caveat:'Success counts are out of 50 custom physical-goal episodes, not official PushT scores. One training seed; equal updates, not equal parameters/FLOPs. Historical depth-12 timing used different hardware/scheduling.'},
 {id:'independent-depth-depth_horizons',chapter:'Depth',title:'Depth and physical forecasting',x:'horizon',metrics:[],filters:{depth:4,readout:'prediction'},description:'Held-out physical forecasts for independently trained encoder depths.',caveat:'Different horizon cohorts; H32 is supported by four episodes. These are distinct trainings, not intermediate taps.'},
 {id:'mechanisms-controls',chapter:'Controls',title:'Does observed history contribute beyond actions?',x:'layer',metrics:['spearman','regret'],filters:{kind:'bilinear',control:'intact',scorer:'native'},description:'Intact history, action-only and shuffled-history fits isolate contributions from observations under the same fitting budget.',caveat:'135 small predictor fits with frozen encoders; matched 1,000-step fits do not prove convergence. Switch controls to inspect all conditions. Candidate ranking does not establish closed-loop success.'},
 {id:'mechanisms-control_horizons',chapter:'Controls',title:'Action controls across forecast horizons',x:'horizon',metrics:[...physical,'episodes'],filters:{layer:'linked',kind:'bilinear',control:'intact'},description:'Physical error of intact, action-only and shuffled-history predictors across time.',caveat:'H32 has four episodes. Shuffled controls use unrelated held-out dataset histories, which adds a donor-distribution caveat.'},
 {id:'matched-intervention-effects',chapter:'Interventions',title:'Physical directions versus matched random directions',x:'layer',metrics:['excess_block_mse'],filters:{factor:'all',mode:'both',strength:1},description:'Excess block-position MSE after removing a physical probe subspace relative to rank- and magnitude-matched random interventions.',caveat:'Error bars are source 95% episode-bootstrap intervals, pointwise and exploratory. An interval spanning zero is unresolved. Directions can be nonlinear/redundant; interventions can be off-manifold.'},
 {id:'mechanisms-interventions',chapter:'Interventions',title:'Token interventions: every condition',x:'layer',metrics:['block_rmse','pusher_rmse','angle_mae','ranking_rho'],filters:{factor:'all',mode:'both',strength:1,control:'physical'},description:'CLS, pooled-patch and joint token interventions, physical subspaces, two doses and random controls.',caveat:'32 test episodes × four frames. The smaller candidate panel has eight banks. Missing ranking measurements are omitted, not plotted as zero.'},
 {id:'full-depth-interventions',chapter:'Interventions',title:'Original CLS-only intervention study',x:'layer',metrics:['physical_minus_random','baseline_block_rmse','physical_block_rmse','random_block_rmse','normalized_displacement_ratio'],filters:{strength:1},description:'Earlier probe-subspace intervention within the full-depth encoder, propagated through remaining blocks while patch tokens stay intact.',caveat:'Original random controls match rank, not intervention magnitude. The later token panel adds magnitude matching; do not treat these designs as interchangeable.'},
 {id:'mechanisms-nuisance',chapter:'Appearance',title:'Appearance sensitivity and physical robustness',x:'layer',metrics:['response','block_rmse','pusher_rmse','angle_mae','block_error_increase','image_mse'],filters:{family:'background_color',strength:.3,replicate:'all'},description:'Eight appearance families × three strengths × two realizations. Compare feature response with the associated physical readout error.',caveat:'32 development / 64 test episodes, four dataset frames each. Response is squared feature change / clean development covariance trace, not information content. The default averages the two realizations; synthetic shifts may be severe.'},
 {id:'mechanisms-visited',chapter:'Visited states',title:'Ranking at states the planner actually visits',x:'scorer',metrics:['spearman','top1_regret','topk_overlap','best_candidate_rank'],filters:{part:'test',behavior:'all',seed:'all',replan:'all'},description:'Every model scores the same candidate banks from histories reached by three learned behavior policies. Compare predicted and actual-future features under both scoring rules.',caveat:'Default test selection: 12 seed clusters, 144 banks, 64 candidates per bank. Points average the selected banks; banks are not independent seed replicates. Scorers are categorical; no interpolation or uncertainty interval is implied.'}
];
const units=key=>key.includes('angle')?'radians':key.includes('image_mse')?'normalized RGB²':key==='excess_block_mse'?'px²':key.includes('rmse')||key==='physical_minus_random'||key==='block_error_increase'?'pixels':key.includes('success')?'episodes / 50':key==='train_hours'?'hours':key.includes('parameters')?'parameters':key==='spearman'||key==='ranking_rho'?'Spearman ρ':key==='horizon'?'steps':'dimensionless / count';
const metricName=k=>({block_position_rmse:'Block position RMSE',pusher_position_rmse:'Pusher position RMSE',angle_mae_rad:'Angle MAE',spearman:'Candidate ranking · Spearman',regret:'Candidate regret',top1_regret:'Selected-candidate regret',D:'Normalized regret reduction',participation_ratio:'Covariance participation ratio',exact_zero_fraction:'Exact-zero fraction',excess_block_mse:'Physical − random block MSE',block_error_increase:'Block RMSE increase over clean',response:'Normalized feature response',final_success:'Final goal success',ever_success:'Ever reached goal',ratio:'Background / physical response'})[k]||nice(k);
const label=k=>/^block_/.test(k)?String(+k.slice(6)):({final_norm:'Norm',projector:'Proj',linked:'Link',predicted_native:'Pred. / native',predicted_physical:'Pred. / physical',exact_native:'True / native',exact_physical:'True / physical'})[k]||String(k);
const layerOrder=Array.from({length:12},(_,i)=>`block_${String(i+1).padStart(2,'0')}`).concat(['final_norm','projector','linked']);
let data,def=defs[0],filtered=[],page=0;
function options(values,selected){return values.map(v=>`<option value="${escape(v)}" ${String(v)===String(selected)?'selected':''}>${v==='all'?'All / mean selected rows':escape(nice(v))}</option>`).join('');}
function setChapter(chapter){
 document.querySelectorAll('.chapter-tabs button').forEach(b=>b.setAttribute('aria-pressed',String(b.textContent===chapter)));
 const group=defs.filter(d=>d.chapter===chapter);$('dataset').innerHTML=group.map(d=>`<option value="${d.id}">${d.title}</option>`).join('');setDataset(group[0].id);
}
function setDataset(id){
 def=defs.find(d=>d.id===id);const rows=data.tables[id];
 if(!def.metrics.length){def.metrics=Object.keys(rows[0]).filter(k=>!['depth','horizon','episodes'].includes(k)&&typeof rows[0][k]==='number');}
 $('metric').innerHTML=def.metrics.map(k=>`<option value="${k}">${metricName(k)}</option>`).join('');
 $('filters').innerHTML=Object.entries(def.filters).map(([key,value])=>{
 const vals=[...new Set(rows.map(r=>r[key]))].filter(v=>v!==undefined&&v!==null);if(!vals.length)return '';
 if(value==='all')vals.unshift('all');const chosen=vals.some(v=>String(v)===String(value))?value:vals[0];
 return `<label>${nice(key)}<select data-filter="${key}">${options(vals,chosen)}</select></label>`;
 }).join('');
 $('filters').querySelectorAll('select').forEach(s=>s.onchange=update);update();
}
function update(){
 page=0;const selections=[...$('filters').querySelectorAll('select')].map(s=>[s.dataset.filter,s.value]);
 filtered=data.tables[def.id].filter(r=>selections.every(([key,value])=>value==='all'||String(r[key])===value));
 $('plot-title').textContent=def.title;$('plot-description').textContent=def.description;$('plot-unit').textContent=units($('metric').value);
 $('chart-caveat').textContent=def.caveat;chart();table();
 const key=def.id==='layers'?'explorer.json':def.id==='matched-intervention-effects'?'mechanisms-summary.json':def.id+'.csv';
 $('download').href='pusht-data/'+key;
}
function chart(){
 const metric=$('metric').value,rows=filtered.filter(r=>Number.isFinite(r[metric]));
 if(!rows.length){$('chart').innerHTML='<p>No recorded values for this selection. Missing measurements are not zero.</p>';$('selection-summary').textContent=`${filtered.length} selected rows · no finite ${metricName(metric)} values`;return;}
 const xs=[...new Set(rows.map(r=>r[def.x]))].sort((a,b)=>def.x==='layer'?layerOrder.indexOf(a)-layerOrder.indexOf(b):typeof a==='number'?a-b:String(a).localeCompare(String(b)));
 const series=arms.map(arm=>({arm,points:xs.map(x=>{
 const rr=rows.filter(r=>r.arm===arm&&r[def.x]===x);if(!rr.length)return null;
 const y=rr.reduce((sum,r)=>sum+r[metric],0)/rr.length;
 const ci=(def.id==='matched-intervention-effects'||(def.id==='full-depth-interventions'&&metric==='physical_minus_random'))&&rr.length===1?[rr[0].ci_low,rr[0].ci_high]:null;
 return {x,y,n:rr.length,ci};})})).filter(s=>s.points.some(Boolean));
 const values=series.flatMap(s=>s.points.filter(Boolean).flatMap(p=>[p.y,...(p.ci||[])]));
 let min=Math.min(0,...values),max=Math.max(0,...values);if(min===max)max=min+1;
 const span=max-min;max+=span*.08;if(min<0)min-=span*.08;
 const w=820,h=365,left=66,right=30,top=20,bottom=75;
 const numeric=xs.every(x=>typeof x==='number');const px=x=>left+(numeric?(x-xs[0])/(xs.at(-1)-xs[0]||1):xs.indexOf(x)/(xs.length-1||1))*(w-left-right);
 const py=y=>h-bottom-(y-min)/(max-min)*(h-top-bottom);
 let svg=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${escape(def.title+'; '+metricName(metric))}"><title>${escape(metricName(metric))}</title>`;
 for(let i=0;i<=4;i++){const y=min+(max-min)*i/4;svg+=`<line x1="${left}" x2="${w-right}" y1="${py(y)}" y2="${py(y)}" stroke="#dce2d8"/><text x="${left-10}" y="${py(y)+4}" text-anchor="end" fill="#627575">${escape(format(Number(y.toPrecision(3))))}</text>`;}
 if(min<0)svg+=`<line x1="${left}" x2="${w-right}" y1="${py(0)}" y2="${py(0)}" stroke="#819587" stroke-dasharray="4 4"/>`;
 xs.forEach(x=>{svg+=`<text x="${px(x)}" y="${h-bottom+24}" text-anchor="middle" fill="#536b68">${escape(label(x))}</text>`;});
 svg+=`<text x="${w/2}" y="${h-13}" text-anchor="middle" fill="#627575">${escape(def.x==='layer'?'Native encoder stage':def.x==='depth'?'Independently trained encoder depth':nice(def.x))}${numeric?'':' · categorical'}</text>`;
 for(const s of series){let path='';let pen=false;for(const p of s.points){if(!p){pen=false;continue;}path+=`${pen?'L':'M'}${px(p.x)},${py(p.y)} `;pen=true;}
 if(def.x!=='scorer')svg+=`<path d="${path}" fill="none" stroke="${colors[s.arm]}" stroke-width="2.5"/>`;
 for(const p of s.points.filter(Boolean)){
 const x=px(p.x),y=py(p.y);if(p.ci)svg+=`<path d="M${x},${py(p.ci[0])}V${py(p.ci[1])}M${x-4},${py(p.ci[0])}h8M${x-4},${py(p.ci[1])}h8" stroke="${colors[s.arm]}" fill="none"/>`;
 const text=`${armNames[s.arm]} · ${label(p.x)} · ${metricName(metric)}: ${format(p.y)}${p.ci?'; 95% interval '+p.ci.map(format).join(' to '):''} · ${p.n} source row${p.n===1?'':'s'}`;
 svg+=`<circle cx="${x}" cy="${y}" r="3.8" fill="${colors[s.arm]}" tabindex="0" aria-label="${escape(text)}"><title>${escape(text)}</title></circle>`;
 }}
 $('chart').innerHTML=svg+'</svg>';$('selection-summary').textContent=`${rows.length} measured rows · ${series.length} methods · ${xs.length} ${def.x==='layer'?'stages':'axis values'} · points = mean of selected rows`;
}
function table(){
 const keys=[...new Set(filtered.flatMap(r=>Object.keys(r)))],size=20,start=page*size;
 $('data-table').innerHTML='<thead><tr>'+keys.map(k=>`<th scope="col">${escape(nice(k))}</th>`).join('')+'</tr></thead><tbody>'+filtered.slice(start,start+size).map(r=>'<tr>'+keys.map(k=>`<td>${escape(format(r[k]??null))}</td>`).join('')+'</tr>').join('')+'</tbody>';
 $('table-count').textContent=filtered.length?`Rows ${start+1}–${Math.min(start+size,filtered.length)} of ${filtered.length}`:'No rows';$('previous').disabled=page===0;$('next').disabled=start+size>=filtered.length;
}
const figureNames={'01_layer_information':'Physical information & covariance geometry','02_sensitivity':'Physical versus appearance sensitivity','03_candidate_ranking':'Layer, predictor and scoring-rule comparison','04_physical_horizons_linked':'Physical forecasting at the linked representation','05_interventions':'Original physical-subspace interventions','06_perturbation_examples':'What the perturbations change','01_depth_decisions':'Independent depth: decisions and goal completion','02_depth_probes':'Independent depth: physical-state probes','03_depth_horizons':'Independent depth: forecasting horizons','action_controls':'Observed history versus action-only controls','interventions':'Magnitude-matched token interventions','nuisance':'Appearance sensitivity and physical readout','nuisance_layers':'Appearance sensitivity through depth','visited':'Scoring at planner-visited states'};
function figures(){const group=$('figure-group').value;document.querySelector('.gallery').innerHTML=data.figures.filter(f=>group==='all'||f.group===group).map(f=>`<figure><a href="pusht-data/${f.file}" target="_blank" rel="noopener"><img loading="lazy" src="pusht-data/${f.file}" alt="${figureNames[f.title]||nice(f.title)}"></a><figcaption>${figureNames[f.title]||nice(f.title)}<small>${nice(f.group)} · original audited figure · open full size ↗</small></figcaption></figure>`).join('');}
document.querySelector('.blocks').innerHTML=Array.from({length:12},(_,i)=>`<span>${i+1}</span>`).join('');
try{
 const response=await fetch('pusht-data/explorer.json');if(!response.ok)throw Error('Evidence files unavailable');data=await response.json();
 for(const r of data.tables['mechanisms-nuisance'])r.block_error_increase=r.block_rmse-r.clean_block;
 document.querySelector('.chapter-tabs').innerHTML=[...new Set(defs.map(d=>d.chapter))].map(c=>`<button aria-pressed="false">${c}</button>`).join('');document.querySelectorAll('.chapter-tabs button').forEach(b=>b.onclick=()=>setChapter(b.textContent));
 $('dataset').onchange=()=>setDataset($('dataset').value);$('metric').onchange=()=>{page=0;chart();table();$('plot-unit').textContent=units($('metric').value);};$('previous').onclick=()=>{page--;table();};$('next').onclick=()=>{page++;table();};$('figure-group').onchange=figures;
 setChapter('Layers');figures();
}catch(e){$('error').textContent='Unable to load the result explorer: '+e.message;$('plot-title').textContent='Evidence unavailable';}
