"""Replot published aggregate curves only; performs no training or inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(source, output):
    data = json.loads(source.read_text())
    output.mkdir(parents=True, exist_ok=True)
    horizons = data['config']['horizons']
    times = np.array(horizons) * data['config']['dt']
    kinds = ['affine', 'mlp64', 'mlp256']
    colors = ['#2d7294', '#bc7c25', '#aa496b']
    names = ['Affine', 'MLP-64', 'MLP-256']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    for metric, label, suffix in [('position_rmse', 'Position RMSE (box units)', 'position'),
                                  ('velocity_rmse', 'Velocity RMSE (box units/s)', 'velocity')]:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
        all_values = []
        for latent, ax in zip([6, 12, 24, 48], axes.flat):
            rows = sorted([e for e in data['encoders'] if
                           (e['method'], e['latent']) == ('gaussian', latent)], key=lambda e: e['seed'])
            assert [e['seed'] for e in rows] == [0, 1, 2]
            for kind, color, name in zip(kinds, colors, names):
                values = np.array([[e['predictors'][kind]['physical']['linear'][str(h)]['all'][metric]
                                    for h in horizons] for e in rows])
                all_values.extend(values.flat)
                for seed in values: ax.plot(times, seed, color=color, alpha=.26, lw=1)
                ax.plot(times, values.mean(0), '-o', color=color, lw=2, ms=3, label=name)
            future = np.array([[e['actual_future']['linear'][str(h)]['all'][metric]
                                for h in horizons] for e in rows]).mean(0)
            persistence = [data['baselines']['horizons']['persistence'][str(h)]['all'][metric]
                           for h in horizons]
            all_values.extend(future); all_values.extend(persistence)
            ax.plot(times, future, '--', color='#887196', label='Actual-future readout (diagnostic)')
            ax.plot(times, persistence, ':', color='#777', label='True-state persistence (privileged)')
            ax.set_xscale('log', base=2); ax.set_yscale('log')
            ax.set_xticks(times, [f'{t:.2f}' for t in times])
            ax.set(title=f'L{latent} = {latent/12:g} × physical-state dimension',
                   xlabel='Time after initial history (seconds; log spacing)', ylabel=label)
            ax.grid(axis='y', alpha=.2)
        limits = (10 ** np.floor(np.log10(min(v for v in all_values if v > 0))),
                  10 ** np.ceil(np.log10(max(all_values))))
        for ax in axes.flat: ax.set_ylim(*limits)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8,
                   bbox_to_anchor=(.5, .035))
        fig.suptitle('Latent size × predictor complexity: ' + label, fontsize=15)
        fig.text(.5, .012, 'Gaussian codes • ridge state readout • 256 test episodes • '
                 'thin: all 3 seeds; thick: arithmetic means • common log error scale; no clipping',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .10, 1, .95))
        fig.savefig(output / f'capacity-time-{suffix}.png', dpi=170)
        fig.savefig(output / f'capacity-time-{suffix}.pdf')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plot(args.source, args.output)
