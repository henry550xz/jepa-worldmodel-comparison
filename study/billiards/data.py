"""Images-only model windows. Evaluation labels stored in a separate directory."""
import json
from pathlib import Path
import numpy as np


class ImageWindows:
    def __init__(self,root,split,history=3,horizon=1):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        if split not in self.manifest['splits'] or history<1 or horizon<1:raise ValueError('invalid split/window')
        self.ids=self.manifest['splits'][split];self.history=history;self.horizon=horizon
        self.windows=self.manifest['config']['frames']-history-horizon+1
        if self.windows<1:raise ValueError('sequence too short')
    def __len__(self):return len(self.ids)*self.windows
    def __getitem__(self,index):
        if index<0 or index>=len(self):raise IndexError(index)
        episode,start=divmod(index,self.windows)
        with np.load(self.root/'observations'/(self.ids[episode]+'.npz'),allow_pickle=False) as f:rgb=f['rgb']
        stop=start+self.history
        return dict(history=rgb[start:stop].copy(),future=rgb[stop:stop+self.horizon].copy())
