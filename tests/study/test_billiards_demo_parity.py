"""Use BILLIARDS_NODE to opt into cross-language physics checks."""
import json,os,subprocess
from pathlib import Path
import numpy as np
import pytest
from study.billiards.physics import Billiards,State


def test_browser_physics_matches_python():
    node=os.environ.get('BILLIARDS_NODE')
    if not node:pytest.skip('Node executable not supplied')
    module=(Path(__file__).resolve().parents[2]/'demos/billiards/physics.mjs').as_uri()
    b=Billiards();states=[]
    for seed in range(10):
        s=b.initial(seed);a,_=b.step(s,2);states.append(dict(q=s.q.tolist(),v=s.v.tolist(),expected=a.q.tolist(),velocity=a.v.tolist()))
    js='import {World} from '+json.dumps(module)+';const states='+json.dumps(states)+';console.log(JSON.stringify(states.map(s=>{let w=new World();w.q=s.q;w.v=s.v;w.step(2);return {q:w.q,v:w.v};})));'
    output=json.loads(subprocess.check_output([node,'--input-type=module','-e',js],text=True))
    for a,s in zip(output,states):np.testing.assert_allclose(a['q'],s['expected'],atol=1e-10);np.testing.assert_allclose(a['v'],s['velocity'],atol=1e-10)
