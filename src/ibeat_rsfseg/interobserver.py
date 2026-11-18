"""Inter-observer variability for RSF masks.

Compares edited masks from two (or more) raters on the same studies:
Dice, HD95, NSD, and volume agreement (ICC(2,1), Bland-Altman, MAE).
Every pair of raters is compared. To benchmark the model, add it as another
rater in RATERS (a database that contains its predictions as a series).

Run it when nobody is editing: the databases are only read, but a half-written
index.json can still cause read errors.
"""

import os
import json
import itertools

import numpy as np
import pandas as pd
import dbdicom as db
from scipy.ndimage import binary_erosion, distance_transform_edt


BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"
SITE = ('Patients', 'Leeds')
TASK = 'rsf_masks'

# name -> (database path, series name). Add the model predictions here if wanted.
RATERS = {
    'obs1': (os.path.join(BUILD, 'rsfseg', 'stage_4_edit', *SITE, TASK),
             f'{TASK}_edited'),
    'obs2': (os.path.join(BUILD, 'rsfseg', 'stage_4_edit_observer2', *SITE, TASK),
             f'{TASK}_edited'),
}

# Studies to use: the frozen subset of observer 2 (None = all studies that
# every rater has).
SUBSET_FILE = os.path.join(BUILD, 'rsfseg', 'stage_4_edit_observer2', *SITE, TASK,
                           'subset.json')

LABELS = {1: 'left', 2: 'right'}
NSD_TOL_MM = 2.0
N_BOOT = 2000
SEED = 42
OUTDIR = os.path.join(BUILD, 'rsfseg', 'interobserver')


# ---------------------------------------------------------------- metrics

def surface(mask):
    return mask & ~binary_erosion(mask)


def surface_distances(a, b, spacing):
    """Directed surface distances a->b and b->a in mm (None if either is empty)."""
    if not a.any() or not b.any():
        return None
    sa, sb = surface(a), surface(b)
    d_to_b = distance_transform_edt(~sb, sampling=spacing)
    d_to_a = distance_transform_edt(~sa, sampling=spacing)
    return d_to_b[sa], d_to_a[sb]


def case_metrics(a, b, spacing):
    """Dice, HD95, NSD and volumes for two boolean masks."""
    va = a.sum() * np.prod(spacing) / 1000.0  # mL
    vb = b.sum() * np.prod(spacing) / 1000.0
    out = {'vol_a': va, 'vol_b': vb, 'dice': np.nan, 'hd95': np.nan, 'nsd': np.nan}
    if not a.any() and not b.any():
        return out  # Dice undefined: both empty (excluded from means)
    out['dice'] = 2.0 * (a & b).sum() / (a.sum() + b.sum())
    d = surface_distances(a, b, spacing)
    if d is not None:
        d_ab, d_ba = d
        out['hd95'] = max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))
        out['nsd'] = ((d_ab <= NSD_TOL_MM).sum() + (d_ba <= NSD_TOL_MM).sum()) / (
            len(d_ab) + len(d_ba))
    return out


def icc_2_1(x):
    """ICC(2,1): two-way random effects, absolute agreement, single measures.
    x has shape (n subjects, k raters)."""
    n, k = x.shape
    gm = x.mean()
    row = x.mean(axis=1, keepdims=True)
    col = x.mean(axis=0, keepdims=True)
    msr = k * ((row - gm) ** 2).sum() / (n - 1)
    msc = n * ((col - gm) ** 2).sum() / (k - 1)
    mse = ((x - row - col + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def boot_ci(fn, n, rng):
    """95% bootstrap CI of fn(indices) over cases."""
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(idx))
        except Exception:
            pass
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if len(vals) else (np.nan, np.nan)


# ---------------------------------------------------------------- data

def open_rater(path, series_name):
    dbd = db.open(path)
    series = {(str(s[1]), s[2][0]): s for s in dbd.series() if s[3][0] == series_name}
    return dbd, series


def load_label(dbd, series):
    vol = dbd.volume(series, None, 0)
    arr = np.round(vol.values).astype(np.int16)
    spacing = np.sqrt((np.asarray(vol.affine)[:3, :3] ** 2).sum(axis=0))
    return arr, spacing


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    raters = {name: open_rater(p, s) for name, (p, s) in RATERS.items()}
    keys = set.intersection(*[set(series) for _, series in raters.values()])
    if SUBSET_FILE and os.path.exists(SUBSET_FILE):
        with open(SUBSET_FILE) as fh:
            keys &= {tuple(k) for k in json.load(fh)}
    keys = sorted(keys)
    print(f"{len(keys)} studies available for all raters: {list(raters)}")

    # Per-case metrics for every rater pair and label
    rows = []
    for key in keys:
        labels = {}
        for name, (dbd, series) in raters.items():
            labels[name] = load_label(dbd, series[key])
        for ra, rb in itertools.combinations(raters, 2):
            (la, sp_a), (lb, sp_b) = labels[ra], labels[rb]
            if la.shape != lb.shape:
                print(f"Skipping {key} {ra}-{rb}: shapes differ {la.shape} vs {lb.shape}")
                continue
            for lab, side in LABELS.items():
                m = case_metrics(la == lab, lb == lab, sp_a)
                rows.append({'patient': key[0], 'study': key[1], 'pair': f"{ra}-{rb}",
                             'side': side, **m})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, 'per_case_metrics.csv'), index=False)

    # Summary per pair and side
    summary = []
    for (pair, side), g in df.groupby(['pair', 'side']):
        g = g.reset_index(drop=True)
        dice = g['dice'].to_numpy()
        hd = g['hd95'].to_numpy()
        nsd = g['nsd'].to_numpy()
        va, vb = g['vol_a'].to_numpy(), g['vol_b'].to_numpy()
        diff = vb - va

        dice_ci = boot_ci(lambda i: np.nanmean(dice[i]), len(g), rng)
        icc = icc_2_1(np.column_stack([va, vb]))
        icc_ci = boot_ci(lambda i: icc_2_1(np.column_stack([va[i], vb[i]])), len(g), rng)

        summary.append({
            'pair': pair, 'side': side, 'n': len(g),
            'dice_mean': np.nanmean(dice), 'dice_ci_low': dice_ci[0], 'dice_ci_high': dice_ci[1],
            'dice_median': np.nanmedian(dice),
            'hd95_median': np.nanmedian(hd),
            'hd95_q1': np.nanpercentile(hd, 25), 'hd95_q3': np.nanpercentile(hd, 75),
            'nsd_mean': np.nanmean(nsd),
            'icc': icc, 'icc_ci_low': icc_ci[0], 'icc_ci_high': icc_ci[1],
            'bias_ml': diff.mean(),
            'loa_low_ml': diff.mean() - 1.96 * diff.std(ddof=1),
            'loa_high_ml': diff.mean() + 1.96 * diff.std(ddof=1),
            'mae_ml': np.abs(diff).mean(),
        })

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUTDIR, 'summary.csv'), index=False)
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(sm.round(3).to_string(index=False))
    print(f"\nSaved to {OUTDIR}")


if __name__ == '__main__':
    main()