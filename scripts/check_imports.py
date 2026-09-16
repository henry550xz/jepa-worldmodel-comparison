"""Import-only validation: no model weights, training, downloads, or GPU kernels."""
import importlib
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
for name in ('torch','torchvision','hydra','decord','train','plan','env.pusht.pusht_wrapper',
             'study.pixel','study.matched_training','study.evaluation','study.probes',
             'study.adapters','study.run','study.readiness_eval','study.reencode_control'):
    importlib.import_module(name)
print('All scientific imports passed; no training performed.')
