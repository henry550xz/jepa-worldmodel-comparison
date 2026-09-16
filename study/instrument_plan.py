"""Measure official planning in-process without changing its objective or search."""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--telemetry',required=True)
    p.add_argument('overrides',nargs=argparse.REMAINDER);a=p.parse_args()
    import torch
    import plan
    if not torch.cuda.is_available():raise RuntimeError('CUDA worker required')
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    sys.argv=['plan.py','--config-path',str(Path(plan.__file__).resolve().parent/'conf')]+(a.overrides[1:] if a.overrides[:1]==['--'] else a.overrides)
    try:plan.main()
    finally:
        torch.cuda.synchronize()
        Path(a.telemetry).write_text(json.dumps({
            'planning_process_wall_seconds':time.perf_counter()-start,
            'planning_peak_vram_bytes':torch.cuda.max_memory_allocated(),
            'planning_peak_reserved_vram_bytes':torch.cuda.max_memory_reserved(),
            'gpu_model':torch.cuda.get_device_name(0),
            'latency_scope':'end-to-end planning including simulator and visualization; not model-only'
        },indent=2)+'\n')

if __name__=='__main__':main()
