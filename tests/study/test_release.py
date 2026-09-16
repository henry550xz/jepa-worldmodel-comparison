"""Portable entry-point and retained-evidence release checks."""
import json
import re
from pathlib import Path
import pytest
from study.run import workspace_root, training_overrides

ROOT=Path(__file__).resolve().parents[2]

def test_workspace_outside_checkout(tmp_path):
    root=workspace_root(tmp_path/'artifacts')
    assert (root/'datasets').is_dir() and (root/'caches/tmp').is_dir()
    with pytest.raises(ValueError):workspace_root(ROOT/'outputs')
    with pytest.raises(ValueError):workspace_root('/')

def test_matched_objective_overrides():
    a=training_overrides('gaussian','pilot',0,'ARTIFACTS','test')
    b=training_overrides('sparse','pilot',0,'ARTIFACTS','test')
    assert set(a)^set(b)=={'link=identity','link=reprelu','target_p=1','target_p=2'}
    for arm in ('pixel','gaussian','sparse'):
        config=training_overrides(arm,'pilot',0,'ARTIFACTS','test')
        assert [x for x in config if x.startswith('training.batch_size=')][-1]=='training.batch_size=32'
        assert 'env.dataset._target_=study.partitions.load_partitioned' in config

def test_preliminary_result_counts():
    pilot={v['method']:v for v in json.loads((ROOT/'results/pilot-summary.json').read_text())}
    assert [pilot[m]['final_success'] for m in ('pixel','gaussian','sparse')]==[0,6,1]
    official=json.loads((ROOT/'results/upstream-reproduction.json').read_text())
    assert [round(x['metrics']['final_eval/success_rate']*50) for x in official]==[36,40]

def test_document_links_resolve():
    for path in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' in link or link.startswith('#'):continue
            assert (path.parent/link.split('#')[0]).exists(),(path.name,link)
