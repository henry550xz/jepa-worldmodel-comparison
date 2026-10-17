// Read-only visual explanations of the selected retained evaluation.
const titles = {pixel: 'Pixel', gaussian: 'Gaussian JEPA', sparse: 'Sparse LpWM'};
const fmt = x => x.toFixed(3);

export function mountDiagnostics(data, metrics, selected) {
  const methods=Object.keys(metrics.methods);
  const root = document.getElementById('diagnostics');
  root.innerHTML = `
    <p class="eyebrow">REPRESENTATION & PREDICTION</p>
    <h2>What the models recover</h2>
    <p>${selected.label} · identical RGB histories across objectives. These diagnostic results do not establish a ranking of objectives.</p>
    <div class="controls">
      <label>Inspect model <select id="diag-method">${methods.map(m => `<option value="${m}">${titles[m]}</option>`).join('')}</select></label>
      <label>Forecast horizon <select id="diag-horizon">${[1,2,4,8,16,32].map(h => `<option value="${h}" ${h===8?'selected':''}>H${h} · ${(h*data.dt).toFixed(2)} s</option>`).join('')}</select></label>
      <label>Projection <select id="diag-camera"><option value="xy">XY</option><option value="xz">XZ</option><option value="yz">YZ</option></select></label>
    </div>
    <div class="diag-grid">
      <article class="diag-card"><h3>1. Trajectory correspondence</h3>
        <p>One held-out episode. Solid colored paths / filled balls: truth. Dashed paths / rings: predicted states from a supervised probe. Numbers identify balls.</p>
        <div id="diag-trajectory"></div><p id="diag-trajectory-note"></p>
      </article>
      <article class="diag-card"><h3>2. Physical-state recovery</h3>
        <p>Position RMSE across held-out episodes; lower is better. The same probe reads either predicted features or features encoded from actual future images.</p>
        <div id="diag-errors"></div>
        <p id="diag-interpretation"></p>
        <p class="note">Large errors with actual images implicate representation and/or probe limitations. A large rollout gap adds predictor/readout incompatibility. This comparison does not isolate the cause.</p>
      </article>
      <article class="diag-card"><h3>3. Visual prediction and object detail</h3>
        <p>Actual saved Pixel output at the selected horizon, regardless of the model selected above. Top: truth. Bottom: predicted RGB. Each row contains XY / XZ / YZ views.</p>
        <img id="diag-rgb" alt="Top row: ground truth balls in three views. Bottom row: Pixel RGB forecast." />
        <p>At this training budget, Pixel predictions are largely uniform and do not recover the visible balls. Foreground-sensitive evaluation is therefore important alongside aggregate image loss.</p>
        <p class="note">The completed pilot still underperforms image persistence. Fitting and readout diagnostics are needed to distinguish optimization, representation and decoder effects.</p>
      </article>
      <article class="diag-card"><h3>4. State validity and event coverage</h3>
        <p>Fraction of estimated initial states satisfying the exact simulator’s constraints.</p><div id="diag-validity"></div>
        <p class="note">State-supervised probe → exact physics control. Rejected states violate simulator constraints. Errors on different valid subsets cannot fairly rank methods.</p>
        <h4>Collision evidence at this horizon</h4><div id="diag-events"></div>
        <p class="note">Counts cover events anywhere from the forecast start to the target. Wall and ball groups may overlap; these are not disjoint bins. Zero means no evidence, not zero error.</p>
      </article>
    </div>
    <p class="diag-takeaway"><strong>Research direction:</strong> Establish learning and state-recovery behavior under matched training budgets, then measure sensitivity to scene complexity, appearance and model depth. The current results motivate these controls; they do not isolate an objective-level limitation.</p>
    <details><summary>Data and evaluation protocol</summary><p>Run: ${selected.run}. Source: <code>${data.source_sha??selected.source_sha??"See retained report"}</code>. Trajectory: ${data.episode??selected.episode??"First retained probe-test episode"}. Charts use <a href="${selected.path}/metrics.json">saved metrics</a> and <a href="${selected.path}/playback.json">saved playback</a>. These analyses use fixed recorded results and remain unchanged when the interactive sandbox is edited. The demo does not run live learned-model inference.</p></details>`;

  if(!methods.includes('pixel'))document.getElementById('diag-rgb').closest('article').remove();

  function update() {
    const method = document.getElementById('diag-method').value;
    const h = +document.getElementById('diag-horizon').value;
    const camera = document.getElementById('diag-camera').value;
    const axes = {xy:[0,1],xz:[0,2],yz:[1,2]}[camera];
    const project = p => [35+270*p[axes[0]]/data.bounds[axes[0]], 25+270*p[axes[1]]/data.bounds[axes[1]]];
    const point = p => project(p).join(',');
    const palette = ['#ba442a','#236da6','#497846','#977034','#8152a1'];
    let shapes = '';
    for (let ball=0;ball<data.truth[0].length;ball++) {
      const truth = data.truth.slice(0,h+1).map(s=>s[ball]);
      const predicted = data.methods[method].slice(0,h).map(s=>s[ball]);
      shapes += `<polyline points="${truth.map(point).join(' ')}" fill="none" stroke="${palette[ball]}" stroke-width="3"/>
        <polyline points="${predicted.map(point).join(' ')}" fill="none" stroke="${palette[ball]}" stroke-width="2" stroke-dasharray="6 4"/>`;
      for (const [state, filled] of [[truth.at(-1),true],[predicted.at(-1),false]]) {
        const [x,y]=project(state);
        shapes += `<circle cx="${x}" cy="${y}" r="${data.radius*270}" fill="${filled?palette[ball]:'none'}" stroke="${palette[ball]}" stroke-width="2" ${filled?'':'stroke-dasharray="4 3"'}/><text x="${x}" y="${y+4}" text-anchor="middle" fill="${filled?'white':palette[ball]}">${ball+1}</text>`;
      }
    }
    const states = data.methods[method][h-1];
    const outside = states.filter(s=>s.slice(0,3).some((v,i)=>v<data.radius||v>data.bounds[i]-data.radius)).length;
    document.getElementById('diag-trajectory').innerHTML = `<svg viewBox="0 0 340 330" role="img" aria-label="${titles[method]} predicted and true ${camera.toUpperCase()} trajectories at H${h}"><defs><clipPath id="diag-box"><rect x="35" y="25" width="270" height="270"/></clipPath></defs><rect x="35" y="25" width="270" height="270" fill="#eef3ed" stroke="#789189"/><g clip-path="url(#diag-box)">${shapes}</g><text x="35" y="16">0</text><text x="290" y="16">1</text><text x="150" y="320">${camera[0].toUpperCase()} →</text><text x="5" y="165">${camera[1].toUpperCase()} ↓</text></svg>`;
    document.getElementById('diag-trajectory-note').textContent = outside ? `${outside}/${states.length} predicted ball centers violate the radius-adjusted 3D walls. Off-box drawings are cropped, not corrected. Raw predicted XYZ: ${states.map((s,i)=>`ball ${i+1} (${s.slice(0,3).map(fmt).join(', ')})`).join('; ')}.` : 'Predicted centers are inside the radius-adjusted walls; that alone does not make the trajectory accurate. This is a 2D projection of 3D motion.';
    // Shared log axis keeps the extreme Pixel failure visible without erasing the other methods.
    const values = methods.flatMap(m=>['predicted','true_future_probe'].map(k=>metrics.methods[m].horizons[h][k].position_rmse));
    const low = Math.floor(Math.log10(Math.min(...values))), high = Math.ceil(Math.log10(Math.max(...values)));
    const width = x => 100*(Math.log10(x)-low)/(high-low||1);
    document.getElementById('diag-errors').innerHTML = `<p class="note">Shared logarithmic scale: ${10**low} to ${10**high} box units. Each power of ten has equal width.</p>` + methods.map(m=>{
      const v=metrics.methods[m].horizons[h];
      return `<div class="diag-error-group ${m===method?'selected':''}"><strong>${titles[m]}</strong>${[['predicted','Predicted features'],['true_future_probe','Actual future features']].map(([k,label])=>`<div class="diag-bar-label">${label}<strong>${fmt(v[k].position_rmse)}</strong></div><div class="diag-track"><span class="${k}" style="width:${width(v[k].position_rmse)}%"></span></div>`).join('')}</div>`;
    }).join('');
    const v=metrics.methods[method].horizons[h];
    document.getElementById('diag-interpretation').textContent = `${titles[method]}: rollout error is ${(v.predicted.position_rmse/v.true_future_probe.position_rmse).toFixed(1)}× the actual-future probe error at H${h}. ${method==='pixel'?'The predicted-feature readout produces coordinates far outside the physical domain at this training budget.':'Compare rollout error with actual-future readout error to assess the additional prediction gap; this does not isolate its cause.'}`;
    const rgb=document.getElementById('diag-rgb');if(rgb)rgb.src=`${selected.path}/pixel-h${h}.png`;
    document.getElementById('diag-validity').innerHTML=methods.map(m=>{
      const r=metrics.methods[m].supervised_image_to_state_to_physics;
      return `<div class="diag-bar-label">${titles[m]}<strong>${r.valid_initial_states}/${r.total} valid</strong></div><div class="diag-valid-track" aria-label="${r.total-r.valid_initial_states} rejected of ${r.total}"><span style="width:${100*r.valid_initial_states/r.total}%"></span></div>`;
    }).join('')+'<p class="note">Green = valid; rust = rejected. Validity is measured at the initial state, so it does not change with horizon.</p>';
    document.getElementById('diag-events').innerHTML=['free','wall','ball'].map(k=>{
      const n=metrics.methods[method].event_strata[k][h].episodes;
      return `<div class="diag-bar-label">${{free:'No collision',wall:'Wall collision',ball:'Ball collision'}[k]}<strong>${n} episodes${n===0?' · no evidence':''}</strong></div><div class="diag-track"><span class="predicted" style="width:${100*n/metrics.methods[method].supervised_image_to_state_to_physics.total}%"></span></div>`;
    }).join('');
  }
  for (const id of ['diag-method','diag-horizon','diag-camera']) document.getElementById(id).addEventListener('change',update);
  update();
}
