"""Scientific-contract tests; tiny fixtures are not reported as research results."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from study.billiards.capacity.model import HistoryEncoder,HistoryJEPA,Predictor,rollout
from study.billiards.capacity.data import Corpus,audit
from study.billiards.capacity.engine import baseline_controls,state_control,image_train,cache_encoder,downstream,supervised_eval,done,fit_features,seed_all
from study.billiards.capacity.report import build
from study.billiards.ablation_data import generate

CONFIG=Path(__file__).resolve().parents[2]/'conf/study/billiards-capacity.json'

def small():
    c=json.loads(CONFIG.read_text());c.update(encoder_width=16,encoder_depth=1,temporal_width=16,latent_sizes=[6],seeds=[0],image_updates=2,image_batch=2,feature_updates=2,feature_batch=8,readout_updates=2,projections=4,validation_every=1,warmup=1,save_updates=[2]);c['condition'].update(height=16,width=48);c['splits']={k:4 for k in c['splits']};return c

class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_dimensions_independent_of_backbone(self):
        c=small();x=torch.randn(2,4,3,16,48)
        for method in ['gaussian','sparse']:
            counts=[]
            for size in [6,12,24,48]:
                m=HistoryJEPA(c,size,method);z=m.encoder(x);self.assertEqual(z.shape,(2,2,size));loss,_=m.loss(x);loss.backward();self.assertTrue(torch.isfinite(loss));counts.append(sum(p.numel() for p in m.encoder.blocks.parameters()))
            self.assertEqual(len(set(counts)),1)
    def test_history_causality_and_rollout(self):
        c=small();m=HistoryEncoder(c,12,'gaussian').eval();x=torch.randn(2,4,3,16,48);y=x.clone();y[:,3]+=10
        with torch.no_grad():self.assertTrue(torch.allclose(m(x)[:,0],m(y)[:,0]))
        net=Predictor(2,0)
        with torch.no_grad():net.net.weight.copy_(torch.eye(2));net.net.bias.fill_(1)
        r=rollout(net,torch.zeros(1,2),3);self.assertTrue(torch.equal(r,torch.tensor([[[1.,1.],[2.,2.],[3.,3.]]])))
    def test_complete_tiny_pipeline_and_integrity(self):
        c=small()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);generate(root/'data',c['condition'],c['splits'],c['frames'],c['dt'],{'git_sha':'fixture'})
            self.assertEqual(audit(root/'data',c)['episodes'],20);corpus=Corpus(root/'data');baseline_controls(corpus,c,root)
            for kind in c['predictors']:state_control(corpus,c,root,kind,0)
            for method in ['gaussian','sparse','supervised']:
                latent=12 if method=='supervised' else 6
                image_train(corpus,c,root,method,latent,0,'cpu');cache_encoder(corpus,c,root,method,latent,0,'cpu')
                if method=='supervised':supervised_eval(corpus,c,root,0)
                else:downstream(corpus,c,root,method,latent,0)
            build(root,c);self.assertTrue((root/'report/state-controls.png').exists());self.assertTrue((root/'report/capacity-linear.png').exists())
            folder=root/'state-controls/affine-seed0';self.assertTrue(done(folder));(folder/'result.json').write_text('{}')
            with self.assertRaises(RuntimeError):done(folder)
    def test_resume_matches_uninterrupted_features(self):
        c=small();c['feature_updates']=4;rng=np.random.default_rng(87)
        x=rng.normal(size=(3,5,6)).astype('float32');y=x*.9+.1
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);seed_all(3);a=Predictor(6,8);fit_features(a,x,y,x,y,root/'full',c,3)
            seed_all(3);b=Predictor(6,8);fit_features(b,x,y,x,y,root/'resume',c,3,steps=2)
            fit_features(b,x,y,x,y,root/'resume',c,3,steps=4)
            for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
    def test_split_overlap_rejected(self):
        c=small()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);ids={k:[f'{k}-{i}' for i in range(4)] for k in c['splits']};ids['test']=ids['world_train'];(root/'manifest.json').write_text(json.dumps({'status':'complete','splits':ids,'files':{}}))
            with self.assertRaises(AssertionError):audit(root,c)

if __name__=='__main__':unittest.main()
