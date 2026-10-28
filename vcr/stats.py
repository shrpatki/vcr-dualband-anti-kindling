import numpy as np
import pandas as pd
from scipy import stats


def cohen_dz(diffs):
    diffs = np.asarray(diffs, float)
    n = len(diffs)
    sd = diffs.std(ddof=1)
    if sd == 0:
        return np.nan, np.nan, np.nan
    dz = diffs.mean() / sd
    se = np.sqrt(1.0 / n + dz ** 2 / (2 * n))
    t = stats.t.ppf(0.975, n - 1)
    return dz, dz - t * se, dz + t * se


def paired(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    p = stats.ttest_rel(x, y).pvalue
    try:
        p_w = stats.wilcoxon(x, y).pvalue if np.any(d != d[0]) else np.nan
    except ValueError:
        p_w = np.nan
    dz, lo, hi = cohen_dz(d)
    return dict(mean_a=x.mean(), mean_b=y.mean(), diff=d.mean(), a_lower=int((d < 0).sum()), n=len(d),
                p=p, p_wilcoxon=p_w, dz=dz, ci_low=lo, ci_high=hi)


def holm(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    order = np.argsort(pvals)
    m = len(pvals)
    significant = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if pvals[i] < alpha / (m - rank):
            significant[i] = True
        else:
            break
    return significant


def by_sim(df, condition, metric):
    return df[df['condition'] == condition].set_index('sim')[metric].sort_index()


def paired_values(df, a, b, metric):
    x, y = by_sim(df, a, metric), by_sim(df, b, metric)
    common = x.index.intersection(y.index)
    return x.loc[common], y.loc[common]


def compare(df, a, b, metric='Cav_post'):
    x, y = paired_values(df, a, b, metric)
    return dict(a=a, b=b, metric=metric, **paired(x, y))


def compare_against(df, reference, others, metric='Cav_post', correct=True):
    table = pd.DataFrame([compare(df, reference, o, metric) for o in others])
    if correct:
        table['holm'] = holm(table['p'].values)
    return table


def friedman(df, metric='Cav_post'):
    table = df.pivot(index='sim', columns='condition', values=metric).dropna()
    result = stats.friedmanchisquare(*[table[c].values for c in table.columns])
    return result.statistic, result.pvalue


def tost(df, a, b, margin, metric='Cav_post'):
    """Two one-sided paired t-tests for equivalence within +- margin; returns the larger p."""
    x, y = paired_values(df, a, b, metric)
    d = (x - y).values
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    p_lower = stats.t.sf((d.mean() + margin) / se, n - 1)
    p_upper = stats.t.cdf((d.mean() - margin) / se, n - 1)
    return max(p_lower, p_upper)


def best(df, metric='Cav_post'):
    return df.groupby('condition')[metric].mean().idxmin()


def split_half(df, metric='Cav_post'):
    """Select the condition with the lowest mean on odd networks and test it on the even ones."""
    odd = df['sim'] % 2 == 1
    selected = best(df[odd], metric)
    test = df[~odd]
    others = [c for c in test['condition'].unique() if c != selected]
    return selected, compare_against(test, selected, others, metric)


def bootstrap_first(df, n_boot=1000, metric='Cav_post', seed=42):
    """Fraction of resamples of networks, keeping pairing, in which each condition has the lowest mean."""
    rng = np.random.default_rng(seed)
    sims = np.array(sorted(df['sim'].unique()))
    m = df.pivot(index='sim', columns='condition', values=metric)
    wins = {c: 0 for c in m.columns}
    for _ in range(n_boot):
        sample = rng.choice(sims, size=len(sims), replace=True)
        wins[m.loc[sample].mean().idxmin()] += 1
    return pd.Series(wins).sort_values(ascending=False) / n_boot


def hold_sensitivity(df, thresholds=(0.80, 0.90, 0.95)):
    cols = {}
    for thr in thresholds:
        held = df['R_post'] < thr * df['R_baseline']
        cols[f'maintained_{int(thr * 100)}'] = held.groupby(df['condition']).mean()
    return pd.DataFrame(cols)


def summarize(df, metrics=('R_on', 'R_post', 'Cav_post', 'Cav_weighted_post')):
    agg = {}
    for m in metrics:
        agg[f'{m}_mean'] = (m, 'mean')
        agg[f'{m}_sd'] = (m, 'std')
    agg['maintained'] = ('held', 'sum')
    agg['n'] = ('sim', 'count')
    return df.groupby('condition').agg(**agg)
