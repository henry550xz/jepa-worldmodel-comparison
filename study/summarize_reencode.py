"""Summarize saved frozen inference controls without fitting or rerunning anything."""
import json,math,argparse,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();root=Path(a.folder);state=json.loads((root/'state.json').read_text());data=json.loads((root/'open-loop.json').read_text());summary={}
 for mode,rows in data.items():
  hs={}
  for h in ['1','2','4','8','16','32']:
   vals=[r['metrics'][h]['prediction'] for r in rows if r['metrics'][h]['status']=='evaluated'];hs[h]={'n':len(vals),**{k:math.sqrt(statistics.mean(v[k]**2 for v in vals)) if k.endswith('rmse') else statistics.mean(v[k] for v in vals) for k in vals[0]}}
  d={'horizons':hs};f=root/f'{mode}-banks.json'
  if f.exists():
   bs=json.loads(f.read_text());d['planning']={'episodes':len(bs),'final_success':sum(b['closed_loop'][-1]['success'] for b in bs),'ever_success':sum(any(t['success'] for t in b['closed_loop']) for b in bs),'final_cost':statistics.mean(b['closed_loop'][-1]['physical_cost'] for b in bs),'regret':statistics.mean(b['fixed_candidate_ranking']['top1_regret'] for b in bs),'spearman':statistics.mean(b['fixed_candidate_ranking']['spearman'] for b in bs)}
  summary[mode]=d
 (root/'summary.json').write_text(json.dumps({'status':state['status'],'conditions':summary},indent=2)+'\n')
 s='# Decode→re-encode control results\n\nStatus: **'+state['status']+'**. No training. Planning results are final only when all three conditions have50 episodes and state is complete. Teacher forcing uses privileged simulator prefixes, not deployable model-only planning.\n\n## Physical errors versus horizon\n\nPusher RMSE / block RMSE / wrapped angle MAE; positions in environment units, angles radians. Same held-out cohorts and frozen linear probe.\n\n| H | N | Original | Decode→re-encode | Teacher-forced oracle |\n|---|---:|---|---|---|\n'
 for h in ['1','2','4','8','16','32']:
  s+=f"|{h}|{summary['original']['horizons'][h]['n']}|"+'|'.join('/'.join(f'{summary[m]["horizons"][h][k]:.2f}' for k in ['pusher_position_rmse','block_position_rmse','angle_mae_rad']) for m in ['original','reencode','teacher_forced_oracle'])+'|\n'
 s+='\n## Planning and candidate ranking\n\n| Mode | Finished episodes /50 | Final success | Ever success | Mean regret | Mean Spearman |\n|---|---:|---:|---:|---:|---:|\n'
 for m,d in summary.items():
  v=d.get('planning')
  if v:s+=f"|{m}|{v['episodes']}|{v['final_success']}|{v['ever_success']}|{v['regret']:.5f}|{v['spearman']:.3f}|\n"
  else:s+=f'|{m}|pending|pending|pending|pending|pending|\n'
 s+='\n## Interpretation limits\n\nH1 repair is a readout change: no predicted latent has yet been fed back. Longer-horizon differences combine readout and feedback changes. Decode/reencode reduces block error at H1–16 but increases it at H32 in this cohort (only4 episodes); do not claim a complete repair or generalized superiority. Teacher forcing supplies fresh true history at every requested endpoint, so it diagnoses accumulation but is not a feasible open-loop substitute. Interpret planning comparisons only when every condition has50 episodes; partial rows are not final results. The original physical metrics were checked against retained pilot outputs. See docs/METHODS.md and manifests/SCIENTIFIC_PROVENANCE.json for settings and original experiment hashes.\n'
 (root/'SUMMARY.md').write_text(s)
 print(s)
if __name__=='__main__':main()
