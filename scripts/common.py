import argparse
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(ROOT, 'figures')
TRACES = 'traces.npz'


def parser(description, sims):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--sims', default=sims, help='network indices, for example 0-29 or 0,3,5')
    p.add_argument('--jobs', type=int, default=None, help='parallel worker processes')
    p.add_argument('--out', default=RESULTS)
    return p


def sim_range(text):
    sims = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-')
            sims.extend(range(int(a), int(b) + 1))
        else:
            sims.append(int(part))
    return sims


def save(df, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    df.to_csv(path, index=False)
    print('wrote', path)


def save_traces(out, entries):
    """Add R and Cav traces to results/traces.npz, keeping the conditions already stored.

    `entries` holds (condition, network index, {t, R, Cav}) tuples.
    """
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, TRACES)
    data = dict(np.load(path)) if os.path.exists(path) else {}
    by_condition = {}
    for condition, sim, trace in entries:
        by_condition.setdefault(condition, []).append((sim, trace))
    for condition, items in by_condition.items():
        items.sort()
        data['t'] = items[0][1]['t'].astype(np.float32)
        data[f'{condition}/sim'] = np.array([sim for sim, _ in items], dtype=np.int16)
        data[f'{condition}/R'] = np.array([trace['R'] for _, trace in items], dtype=np.float32)
        data[f'{condition}/Cav'] = np.array([trace['Cav'] for _, trace in items], dtype=np.float32)
    np.savez_compressed(path, **data)
    print('wrote', path)


def load_traces(results, condition):
    """Time axis and the R and Cav traces of one condition, one row per network."""
    data = np.load(os.path.join(results, TRACES))
    return data['t'], data[f'{condition}/sim'], data[f'{condition}/R'], data[f'{condition}/Cav']
