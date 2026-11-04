"""Figures use retained results only; pending cells are never numeric zeros."""
from pathlib import Path
import json
import numpy as np
from .data import write


def build(root,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf9f4','axes.facecolor':'#faf9f4','savefig.facecolor':'#faf9f4'})
    rows=[]
    for p in sorted((root/'state-controls').glob('*/result.json')):
        d=json.loads(p.read_text())
        if 'kind' in d:rows.append(d)
    if rows:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');colors={'affine':'#c06c50','mlp64':'#347f9e','mlp256':'#579b75'}
        for ax,metric in zip(axes,['position_rmse','velocity_rmse']):
            for kind in cfg['predictors']:
                values=np.array([[r['horizons'][str(h)]['all'][metric] for h in cfg['horizons']] for r in rows if r['kind']==kind])
                if not len(values):continue
                for v in values:ax.plot(cfg['horizons'],v,color=colors[kind],alpha=.25,lw=1)
                ax.plot(cfg['horizons'],values.mean(0),label=kind,color=colors[kind],lw=2,marker='o')
            baseline=root/'state-controls/baselines/result.json'
            if baseline.exists():
                d=json.loads(baseline.read_text())
                for label,style in [('constant_velocity','--'),('persistence',':'),('exact_physics','-.')]:ax.plot(cfg['horizons'],[d['horizons'][label][str(h)]['all'][metric] for h in cfg['horizons']],style,label=label,color='#777777',alpha=.8)
            ax.set(xlabel='Forecast horizon (steps; 0.04 s each)',ylabel=metric.replace('_',' '));ax.legend(fontsize=8)
        fig.suptitle('True-state controls — privileged position/velocity inputs\nThin lines: training seeds; thick lines: mean. Not an image-model result.',fontsize=12)
        fig.savefig(out/'state-controls.png',dpi=180);fig.savefig(out/'state-controls.pdf');plt.close(fig)
    result_rows=[]
    for p in sorted((root/'downstream').glob('*/*/result.json')):
        d=json.loads(p.read_text())
        if 'physical' in d:result_rows.append(d)
    # Full scientific heatmaps exist only when their numerical cells exist.
    if result_rows:
        for readout in ['linear','mlp']:
            fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
            for col,method in enumerate(cfg['methods']):
                for row,metric in enumerate(['position_rmse','velocity_rmse']):
                    a=np.full((len(cfg['predictors']),len(cfg['latent_sizes'])),np.nan)
                    for i,kind in enumerate(cfg['predictors']):
                        for j,l in enumerate(cfg['latent_sizes']):
                            vals=[d['physical'][readout]['predicted']['32']['all'][metric] for d in result_rows if d['method']==method and d['predictor']==kind and d['latent']==l]
                            if len(vals)==len(cfg['seeds']):a[i,j]=np.mean(vals)
                    ax=axes[row,col];ax.set_title(method+' · '+metric);im=ax.imshow(a,cmap='viridis_r',aspect='auto');ax.set_xticks(range(len(cfg['latent_sizes'])),cfg['latent_sizes']);ax.set_yticks(range(len(cfg['predictors'])),cfg['predictors']);ax.set_xlabel('History latent dimension (physical state = 12)')
                    for i in range(a.shape[0]):
                        for j in range(a.shape[1]):ax.text(j,i,f'{a[i,j]:.3f}' if np.isfinite(a[i,j]) else 'pending',ha='center',va='center',color='white' if np.isfinite(a[i,j]) else 'black')
                    fig.colorbar(im,ax=ax)
            # Consistent scale across objectives, separately for position/velocity.
            for row in range(2):
                images=[axes[row,col].images[0] for col in range(2)];vals=np.concatenate([im.get_array().compressed() for im in images]);hi=float(vals.max()) if len(vals) else 1
                for im in images:im.set_clim(0,max(hi,1e-10))
            fig.suptitle('H32 physical forecasting · '+readout+' readout · seed means');fig.savefig(out/f'capacity-{readout}.png',dpi=180);fig.savefig(out/f'capacity-{readout}.pdf');plt.close(fig)
    if result_rows:capacity_detail_figures(root,cfg,result_rows)
    coverage=[]
    for method in cfg['methods']:
        for l in cfg['latent_sizes']:
            for seed in cfg['seeds']:
                name=f'{method}-L{l}-seed{seed}'
                coverage.append(dict(method=method,latent=l,seed=seed,image='complete' if (root/'images'/name/'receipt.json').exists() else 'unrun',predictors={k:'complete' if (root/'downstream'/name/k/'receipt.json').exists() else 'unrun' for k in cfg['predictors']}))
    write(out/'coverage.json',coverage);write(out/'state-control-results.json',rows);write(out/'capacity-results.json',result_rows)
    lines=['# Capacity experiment results','',f'Completed true-state predictor fits: {len(rows)}/{len(cfg["seeds"])*len(cfg["predictors"])}.',f'Completed image-based downstream cells: {len(result_rows)}/{len(coverage)*len(cfg["predictors"])}.','', 'True-state controls use privileged labels. They do not demonstrate learned visual representations. Unrun image cells are not failures and have no numerical result.','', '| Predictor | Seed | H32 position RMSE | H32 velocity RMSE |','|---|---:|---:|---:|']
    for d in rows:
        m=d['horizons']['32']['all'];lines.append(f'| {d["kind"]} | {d["seed"]} | {m["position_rmse"]:.6f} | {m["velocity_rmse"]:.6f} |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def capacity_detail_figures(root,cfg,rows):
    """Companion B/C panels and paired seed-level non-inferiority intervals."""
    import matplotlib.pyplot as plt
    from scipy.stats import t as student_t
    out=Path(root)/'report';decisions=[]
    for readout in ['linear','mlp']:
        fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        for col,method in enumerate(cfg['methods']):
            for row,metric in enumerate(['position_rmse','velocity_rmse']):
                ax=axes[row,col]
                for k,kind in enumerate(cfg['predictors']):
                    group=[d for d in rows if d['method']==method and d['predictor']==kind]
                    for source,style in [('predicted','-'),('actual_future','--')]:
                        means=[]
                        for l in cfg['latent_sizes']:
                            values=[d['physical'][readout][source]['32']['all'][metric] for d in group if d['latent']==l]
                            means.append(np.mean(values) if len(values)==len(cfg['seeds']) else np.nan)
                            ax.scatter([l/12]*len(values),values,s=14,color=f'C{k}',alpha=.35)
                        ax.plot(np.array(cfg['latent_sizes'])/12,means,style,color=f'C{k}',label=kind+' '+source)
                    ref={d['seed']:d for d in group if d['latent']==max(cfg['latent_sizes'])}
                    for l in cfg['latent_sizes'][:-1]:
                        pairs=[(d,ref[d['seed']]) for d in group if d['latent']==l and d['seed'] in ref]
                        if len(pairs)!=len(cfg['seeds']) or len(pairs)<2:continue
                        ratios=np.array([d['physical'][readout]['predicted']['32']['all'][metric]/r['physical'][readout]['predicted']['32']['all'][metric] for d,r in pairs])
                        logs=np.log(ratios);delta=student_t.ppf(.95,len(logs)-1)*logs.std(ddof=1)/np.sqrt(len(logs));upper=float(np.exp(logs.mean()+delta))
                        decisions.append(dict(method=method,predictor=kind,latent=l,readout=readout,metric=metric,paired_seed_ratios=ratios.tolist(),one_sided_95_upper=upper,margin=1+cfg['noninferiority_margin'],relative_noninferior=upper<=1+cfg['noninferiority_margin'],caveat='Exploratory paired log-ratio t interval across three seeds, no multiplicity correction. Not an absolute usefulness claim; both metrics must satisfy margin.'))
                ax.axvline(1,color='#555',ls=':',lw=1);ax.set(xlabel='Latent dimension / physical state dimension',ylabel=metric,title=method);ax.legend(fontsize=6)
        fig.suptitle('Where error enters · '+readout+' readout\nSolid: predicted rollout; dashed: actual-future readout; dots: seeds')
        fig.savefig(out/f'readout-vs-rollout-{readout}.png',dpi=180);fig.savefig(out/f'readout-vs-rollout-{readout}.pdf');plt.close(fig)
        for stratum in ['all','free','wall','ball']:
            fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
            for col,method in enumerate(cfg['methods']):
                for row,metric in enumerate(['position_rmse','velocity_rmse']):
                    ax=axes[row,col]
                    # Prespecified state-sized L=12; no test-set model selection.
                    for k,kind in enumerate(cfg['predictors']):
                        group=[d for d in rows if d['method']==method and d['latent']==12 and d['predictor']==kind]
                        if len(group)!=len(cfg['seeds']):continue
                        a=np.array([[d['physical'][readout]['predicted'][str(h)][stratum][metric] if d['physical'][readout]['predicted'][str(h)][stratum][metric] is not None else np.nan for h in cfg['horizons']] for d in group],float)
                        for seed_values in a:ax.plot(cfg['horizons'],seed_values,color=f'C{k}',alpha=.2)
                        ax.plot(cfg['horizons'],a.mean(0),color=f'C{k}',label=kind)
                        refs=[]
                        for seed in cfg['seeds']:
                            p=Path(root)/'state-controls'/f'{kind}-seed{seed}'/'result.json'
                            if p.exists():
                                d=json.loads(p.read_text());refs.append([d['horizons'][str(h)][stratum][metric] for h in cfg['horizons']])
                        if len(refs)==len(cfg['seeds']):ax.plot(cfg['horizons'],np.array(refs,dtype=float).mean(0),'--',color=f'C{k}',label=kind+' true-state')
                    ax.set(xlabel='Forecast steps (0.04 s)',ylabel=metric,title=method)
                    if ax.lines:ax.legend(fontsize=7)
            fig.suptitle(f'Prespecified L=12 horizon comparison · {stratum} events · {readout} readout')
            fig.savefig(out/f'horizons-{stratum}-{readout}.png',dpi=180);plt.close(fig)
    write(out/'paired-noninferiority.json',decisions)
