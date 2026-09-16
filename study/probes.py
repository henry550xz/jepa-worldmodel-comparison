"""Frozen-feature linear/MLP probes; PyTorch is required only on worker."""
import copy
import torch
from torch import nn


def encode_frozen(model, observations):
    model.eval()
    with torch.no_grad():
        return model.encode_obs_linked(observations)['visual'].flatten(start_dim=2).detach()


def state_targets(states):
    return torch.cat((states[..., :4], torch.sin(states[..., 4:5]),
                      torch.cos(states[..., 4:5]), states[..., 5:]), dim=-1)


def physical_states(targets):
    return torch.cat((targets[..., :4], torch.atan2(targets[..., 4:5], targets[..., 5:6]),
                      targets[..., 6:]), dim=-1)


class Probe(nn.Module):
    def __init__(self, train_features, train_targets, hidden=0):
        super().__init__()
        x, y = train_features.detach(), train_targets.detach()
        self.register_buffer('x_mean', x.mean(0))
        std=x.std(0, unbiased=False)
        self.register_buffer('x_active', (std>1e-6).to(x.dtype))
        self.register_buffer('x_scale', torch.where(std>1e-6,std,torch.ones_like(std)))
        self.register_buffer('y_mean', y.mean(0))
        self.register_buffer('y_scale', y.std(0, unbiased=False).clamp_min(1e-6))
        self.net = nn.Linear(x.shape[-1], y.shape[-1]) if hidden == 0 else nn.Sequential(
            nn.Linear(x.shape[-1], hidden), nn.ReLU(), nn.Linear(hidden, y.shape[-1]))
        self.to(x.device)

    def forward(self, x):
        return self.net(((x.detach()-self.x_mean)/self.x_scale)*self.x_active)*self.y_scale+self.y_mean


def fit_probe(train_x, train_y, val_x, val_y, *, hidden=0, steps=500, lr=1e-3, seed=0):
    """Caller supplies disjoint episode partitions; test set is never accepted here."""
    if steps < 1 or len(train_x) < 2 or len(val_x) < 1:
        raise ValueError('nonempty train/validation data and positive steps required')
    torch.manual_seed(seed)
    probe = Probe(train_x, train_y, hidden)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    best, best_loss = None, float('inf')
    for _ in range(steps):
        probe.train()
        opt.zero_grad()
        loss = (((probe(train_x)-train_y.detach())/probe.y_scale)**2).mean()
        loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            score = (((probe(val_x)-val_y.detach())/probe.y_scale)**2).mean().item()
        if score < best_loss:
            best_loss, best = score, copy.deepcopy(probe.state_dict())
    if best is None:
        raise ValueError('probe validation loss was nonfinite')
    probe.load_state_dict(best)
    probe.requires_grad_(False)
    return probe.eval(), {'validation_normalized_mse': best_loss, 'seed': seed, 'steps': steps}
