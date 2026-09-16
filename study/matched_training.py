"""Matched train/validation loop, without upstream image/rollout diagnostics.

Same upstream initialized model and optimizers. Timed steady steps exclude the
first two warmup updates. Validation runs in no_grad with the same batch size.
"""
import itertools
import time
import torch


def install(train, telemetry, bounded_steps):
    def run_epoch(self, training):
        self.model.train(training)
        loader=self.dataloaders['train' if training else 'valid']
        limit=(bounded_steps or None) if training else (2 if bounded_steps else None)
        print(f'MATCHED epoch={self.epoch} split={"train" if training else "validation"} total_steps={len(loader)}',flush=True)
        for i,(obs,act,state) in enumerate(itertools.islice(loader,limit)):
            torch.cuda.synchronize();start=time.perf_counter()
            opts=[getattr(self,n,None) for n in ['encoder_optimizer','predictor_optimizer','action_encoder_optimizer','decoder_optimizer']]
            if training:
                for opt in opts:
                    if opt is not None:opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                *_,loss,components=self.model(obs,act)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite matched loss')
                if training:
                    self.accelerator.backward(loss)
                    if i in (0,2):
                        grads={n:sum(float(p.grad.abs().sum()) for p in getattr(self,n).parameters() if p.grad is not None)
                               for n in ['encoder','predictor','action_encoder']+(['decoder'] if self.cfg.has_decoder and getattr(self.model,'mechanism',None)!='gaussian' else [])}
                        required=[v for k,v in grads.items() if k!='action_encoder' or i>=2]
                        if any(not (v>0) for v in required):raise RuntimeError('zero module gradient')
                        telemetry.setdefault('gradient_checks',{})[str(i)]=grads
                        telemetry['gradient_l1']=grads
                    for opt in opts:
                        if opt is not None:opt.step()
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            telemetry.setdefault('train_step_seconds' if training else 'validation_step_seconds',[]).append(elapsed)
            telemetry.setdefault('train_losses' if training else 'validation_losses',[]).append(float(loss.detach()))
            print(f'MATCHED {"train" if training else "validation"} epoch={self.epoch} step={i+1} batch={len(act)} seconds={elapsed:.4f} loss={float(loss.detach()):.6g}',flush=True)
            self.logs_update({('train_' if training else 'val_')+'loss':[float(loss.detach())]})
    train.Trainer.train=lambda self:run_epoch(self,True)
    train.Trainer.val=lambda self:run_epoch(self,False)
