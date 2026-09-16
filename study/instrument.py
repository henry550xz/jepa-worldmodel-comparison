"""Worker-only upstream training instrumentation; reproduction loss is untouched."""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--telemetry', required=True)
    p.add_argument('--smoke-batches', type=int, default=0)
    p.add_argument('--matched', action='store_true')
    p.add_argument('overrides', nargs=argparse.REMAINDER)
    a = p.parse_args()
    import torch
    import train
    if not torch.cuda.is_available():
        raise RuntimeError('GPU training is prohibited on the controller; CUDA worker required')
    matched_telemetry={}
    if a.matched:
        from study.matched_training import install
        install(train,matched_telemetry,a.smoke_batches)
    if a.smoke_batches and not a.matched:
        original = train.Trainer.__init__
        def bounded(self, cfg):
            original(self, cfg)
            self.dataloaders = {key: list(itertools.islice(loader, a.smoke_batches))
                                for key, loader in self.dataloaders.items()}
        train.Trainer.__init__ = bounded
        # Bound smoke validation too; official open-loop diagnostics can be expensive.
        train.Trainer.openloop_rollout = lambda *args, **kwargs: {}
    counts = {}
    original_models = train.Trainer.init_models
    def counted(self):
        original_models(self)
        for name in ('encoder','predictor','action_encoder','decoder'):
            module = getattr(self, name, None)
            counts[name] = sum(p.numel() for p in module.parameters()) if module is not None else 0
    train.Trainer.init_models = counted
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    sys.argv = ['train.py', '--config-path', str(Path(train.__file__).resolve().parent/'conf')] + (a.overrides[1:] if a.overrides[:1] == ['--'] else a.overrides)
    try:
        train.main()
    finally:
        torch.cuda.synchronize()
        Path(a.telemetry).write_text(json.dumps({
            **matched_telemetry,
            'peak_vram_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_vram_bytes': torch.cuda.max_memory_reserved(),
            'gpu_model': torch.cuda.get_device_name(0),
            'training_process_wall_seconds': time.perf_counter()-start,
            'parameter_counts': counts,
            'smoke_batches_per_split': a.smoke_batches,
        }, indent=2)+'\n')

if __name__ == '__main__': main()
