"""Summarise the cross-validation results of the RSF model (stage 10 / 11).

Reads the per-fold validation summaries that nnU-Net wrote at the end of
training (fold_X/validation/summary.json); these are the out-of-fold
predictions compared with the reference (edited) masks. Optionally (--surface)
it also computes HD95 and normalized surface Dice (NSD) from the
cross-validation predictions of stage 11.

For every fold and for all folds pooled, and for the left and right class
separately, it reports:
    - Dice (and NSD, HD95 with --surface)
    - agreement of the RSF volume: Pearson r, ICC(2,1) (two-way random,
      absolute agreement, single measures), Bland-Altman bias and 95% limits
      of agreement, mean absolute error and mean absolute percentage error
with 95% confidence intervals from a cluster bootstrap on patient level (all
scans of a patient are drawn together, because scans of one patient are not
independent). Patient id = first two parts of the case id, e.g. 1128_001 from
1128_001_Baseline and 1128_C01 from 1128_C01_Visit1.

Scopes are: every fold, every scanner group (see scanner_of) and pooled.

Output (in rsfseg/results_summary):
    results_summary.xlsx     all tables in one Excel file (needs openpyxl), one sheet each:
                             Population_site, Population_scanner, Scanner_by_fold,
                             Results, Results_long, Descriptives, Voxel_spacing,
                             Data_quality, Worst_cases, Per_case
    data_quality.csv         empty masks, Dice < 0.7 / 0.5, cases below the % threshold
    worst_cases.csv          lowest-Dice cases per class (all, and reference >= threshold),
                             and with --surface the cases with the highest HD95
    descriptives.csv         mean, SD, median, IQR, min, max per scope, class and metric
                             (the median (IQR) is also added to the Results table)
    voxel_spacing.csv        voxel sizes per scanner group; with --surface a warning is
                             printed if the NSD tolerance is below the in-plane voxel size
    population_site.csv     participants (examinations) per site and cohort
    population_scanner.csv   participants (examinations) per scanner group and cohort
    population_scanner_by_fold.csv  participants per fold and scanner group
    dice_by_scanner.png      boxplot of Dice per case, per scanner group
    results_long.csv         scope, class, metric, estimate, ci_low, ci_high
    results_table.csv        the same, one formatted cell per metric
    dice_per_case.csv        per-case Dice, volumes and (optional) surface metrics
    surface_per_case.csv     HD95 and NSD per case (only with --surface)
    dice_by_fold.png         boxplot of Dice per case, per fold
    dice_by_group.png        boxplot of Dice per case, per case-id prefix (site)
    volume_agreement.png     predicted vs reference volume + Bland-Altman (mL)
    volume_agreement_rel.png Bland-Altman on percentage differences

Volumes come from the voxel counts in the nnU-Net summary and the voxel size
in the label files of stage 8. Adapt SITE_NAMES to give the prefixes a name.
"""

import os
import json
import logging
import argparse

import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PIPELINE = 'rsfseg'
DATASET = 'Dataset001_iBEAtRSF'
TRAINER = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
CROSSVAL = 'crossval_results_folds_0_1_2_3_4'
N_FOLDS = 5
CLASS_MAP = {'1': 'left', '2': 'right'}

# Percentage errors (bias %, MAPE, relative Bland-Altman) are only computed for cases with
# a reference volume of at least this many mL: with a tiny reference volume a small absolute
# error gives an enormous percentage. Decide this before looking at results and report it.
MIN_REF_ML = 1.0

# Case-id prefix -> site (2128 and 6128 are both Bordeaux)
SITE_NAMES = {'1128': 'Bari', '2128': 'Bordeaux', '6128': 'Bordeaux', '3128': 'Exeter',
              '4128': 'Leeds', '5128': 'Turku', '7128': 'Sheffield'}

# Scanner groups, as given by the study team:
#   Siemens (same model and software version): Exeter, Leeds, Bordeaux
#   Philips: Bari, and the first 5 numbered participants of Sheffield
#   Philips, old hybrid system: the first 25 numbered patients of Turku
#   GE PET-MR: all other participants of Turku (controls included) and Sheffield
SIEMENS_PREFIXES = ('2128', '6128', '3128', '4128')
TURKU_PHILIPS_MAX = 25      # patient numbers 1..25 in Turku (5128_001 .. 5128_025)
SHEFFIELD_PHILIPS_MAX = 5   # patient numbers 1..5 in Sheffield (7128_001 .. 7128_005)


def scanner_of(case):
    """Scanner group of a case, from its id (e.g. 5128_012_Baseline)."""
    parts = case.split('_')
    prefix = parts[0]
    number = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None  # controls: C01
    if prefix in SIEMENS_PREFIXES:
        return 'Siemens'
    if prefix == '1128':
        return 'Philips'
    if prefix == '7128':
        return 'Philips' if number is not None and number <= SHEFFIELD_PHILIPS_MAX else 'GE PET-MR'
    if prefix == '5128':
        return 'Philips (old hybrid)' if number is not None and number <= TURKU_PHILIPS_MAX else 'GE PET-MR'
    logging.warning(f'{case}: unknown site prefix {prefix}, scanner set to Unknown')
    return 'Unknown'


def cohort_of(patient):
    """Controls have a C in the second part of the id (1128_C01), patients a number."""
    parts = patient.split('_')
    return 'Controls' if len(parts) > 1 and parts[1].startswith('C') else 'Patients'


# ----------------------------------------------------------------------------
# Reading the nnU-Net summaries
# ----------------------------------------------------------------------------

def case_id(path):
    name = os.path.basename(path)
    return name[:-7] if name.endswith('.nii.gz') else os.path.splitext(name)[0]


def patient_id(case):
    """1128_001_Baseline -> 1128_001 ; 1128_C01_Visit1 -> 1128_C01."""
    return '_'.join(case.split('_')[:2])


def read_fold(train_dir, fold):
    """Return the per-case rows of one fold from nnU-Net's validation summary."""
    path = os.path.join(train_dir, f'fold_{fold}', 'validation', 'summary.json')
    with open(path) as fp:
        summary = json.load(fp)
    rows = []
    for item in summary['metric_per_case']:
        case = case_id(item['prediction_file'])
        group = case.split('_')[0]
        row = {'fold': fold, 'case': case, 'patient': patient_id(case),
               'group': SITE_NAMES.get(group, group), 'scanner': scanner_of(case)}
        for key, name in CLASS_MAP.items():
            m = item['metrics'][key]
            row[f'dice_{name}'] = m['Dice']
            row[f'n_ref_{name}'] = m['n_ref']
            row[f'n_pred_{name}'] = m['n_pred']
        rows.append(row)
    return rows


def voxel_spacing(labels_dir, case):
    """Voxel size (mm) of the label file, read from its header."""
    path = os.path.join(labels_dir, f'{case}.nii.gz')
    return tuple(float(z) for z in nib.load(path).header.get_zooms()[:3])


# ----------------------------------------------------------------------------
# Surface metrics
# ----------------------------------------------------------------------------

def surface(mask):
    """Boundary voxels of a binary mask."""
    return mask & ~ndi.binary_erosion(mask)


def surface_metrics(pred, ref, spacing, tolerance):
    """HD95 (mm) and normalized surface Dice at the given tolerance (mm).

    Returns (nan, nan) if either mask is empty.
    """
    pred, ref = pred.astype(bool), ref.astype(bool)
    if not pred.any() or not ref.any():
        return np.nan, np.nan

    s_pred, s_ref = surface(pred), surface(ref)
    # Distance from every voxel to the nearest surface voxel of the other mask
    dist_to_ref = ndi.distance_transform_edt(~s_ref, sampling=spacing)
    dist_to_pred = ndi.distance_transform_edt(~s_pred, sampling=spacing)
    d_pred = dist_to_ref[s_pred]  # pred surface -> ref surface
    d_ref = dist_to_pred[s_ref]   # ref surface -> pred surface

    hd95 = max(np.percentile(d_pred, 95), np.percentile(d_ref, 95))
    nsd = ((d_pred <= tolerance).sum() + (d_ref <= tolerance).sum()) / (d_pred.size + d_ref.size)
    return float(hd95), float(nsd)


def compute_surface(cases, labels_dir, crossval_dir, tolerance):
    rows = []
    for i, case in enumerate(cases, 1):
        print(f'Surface metrics {i}/{len(cases)}: {case}', flush=True)
        ref_img = nib.load(os.path.join(labels_dir, f'{case}.nii.gz'))
        pred_img = nib.load(os.path.join(crossval_dir, f'{case}.nii.gz'))
        ref = np.rint(ref_img.get_fdata()).astype(np.int16)
        pred = np.rint(pred_img.get_fdata()).astype(np.int16)
        if ref.shape != pred.shape:
            logging.error(f'{case}: shape mismatch {pred.shape} vs {ref.shape}, skipped')
            continue
        spacing = ref_img.header.get_zooms()[:3]
        row = {'case': case}
        for key, name in CLASS_MAP.items():
            hd95, nsd = surface_metrics(pred == int(key), ref == int(key), spacing, tolerance)
            row[f'hd95_mm_{name}'] = hd95
            row[f'nsd_{name}'] = nsd
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Agreement statistics
# ----------------------------------------------------------------------------

def icc_2_1(data):
    """ICC(2,1): two-way random effects, absolute agreement, single measures.

    data: array (n subjects, k raters). Shrout & Fleiss (1979).
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    if n < 3:
        return np.nan
    grand = data.mean()
    ss_rows = k * ((data.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((data.mean(axis=0) - grand) ** 2).sum()
    ss_total = ((data - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return (msr - mse) / denom if denom > 0 else np.nan


def agreement_stats(ref, pred, min_ref_ml=MIN_REF_ML):
    """Agreement between reference and predicted volumes (arrays of equal length).

    Absolute measures use all cases; the percentage measures (bias_pct, mape_pct)
    only use cases with a reference volume of at least min_ref_ml.
    """
    ok = np.isfinite(ref) & np.isfinite(pred)
    ref, pred = ref[ok], pred[ok]
    out = {}
    if len(ref) < 3:
        return {k: np.nan for k in ['pearson', 'icc', 'bias_ml', 'loa_low_ml', 'loa_high_ml',
                                    'mae_ml', 'bias_pct', 'mape_pct']}
    diff = pred - ref
    sd = diff.std(ddof=1)
    out['pearson'] = np.corrcoef(ref, pred)[0, 1] if ref.std() > 0 and pred.std() > 0 else np.nan
    out['icc'] = icc_2_1(np.column_stack([ref, pred]))
    out['bias_ml'] = diff.mean()
    out['loa_low_ml'] = diff.mean() - 1.96 * sd
    out['loa_high_ml'] = diff.mean() + 1.96 * sd
    out['mae_ml'] = np.abs(diff).mean()
    nz = ref >= min_ref_ml
    out['bias_pct'] = (100 * diff[nz] / ref[nz]).mean() if nz.any() else np.nan
    out['mape_pct'] = (100 * np.abs(diff[nz]) / ref[nz]).mean() if nz.any() else np.nan
    return out


def cluster_bootstrap(patients, stat_fn, n_boot, rng):
    """Point estimate and 95% percentile CI, drawing whole patients.

    patients: array with the patient id of every row.
    stat_fn:  function(row_indices) -> dict of statistics.
    """
    codes, uniq = pd.factorize(pd.Series(patients))
    order = np.argsort(codes, kind='stable')
    splits = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    n_pat = len(uniq)

    point = stat_fn(np.arange(len(codes)))
    draws = {k: [] for k in point}
    for _ in range(n_boot):
        picked = rng.integers(0, n_pat, n_pat)
        idx = np.concatenate([splits[i] for i in picked])
        res = stat_fn(idx)
        for k, v in res.items():
            draws[k].append(v)

    result = {}
    for k, v in draws.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 0.9 * n_boot:
            logging.warning(f'{k}: only {len(v)} of {n_boot} bootstrap samples were usable')
        lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
        result[k] = (point[k], lo, hi)
    return result, n_pat


def summarise(df, n_boot, seed, with_surface, min_ref_ml=MIN_REF_ML):
    """Statistics with patient-level bootstrap CIs, per fold and pooled."""
    rng = np.random.default_rng(seed)
    scopes = [(f'fold_{f}', df[df['fold'] == f]) for f in sorted(df['fold'].unique())]
    scopes += [(f'scanner: {s}', df[df['scanner'] == s]) for s in sorted(df['scanner'].unique())]
    scopes.append(('pooled', df))

    rows = []
    for scope, sub in scopes:
        sub = sub.reset_index(drop=True)
        for name in CLASS_MAP.values():
            ref = sub[f'vol_ref_ml_{name}'].to_numpy(float)
            pred = sub[f'vol_pred_ml_{name}'].to_numpy(float)
            cols = {'dice': sub[f'dice_{name}'].to_numpy(float)}
            if with_surface:
                cols['nsd'] = sub[f'nsd_{name}'].to_numpy(float)
                cols['hd95_mm'] = sub[f'hd95_mm_{name}'].to_numpy(float)

            def stat_fn(idx, ref=ref, pred=pred, cols=cols):
                out = {k: np.nanmean(v[idx]) if np.isfinite(v[idx]).any() else np.nan
                       for k, v in cols.items()}
                out.update(agreement_stats(ref[idx], pred[idx], min_ref_ml))
                return out

            res, n_pat = cluster_bootstrap(sub['patient'].to_numpy(), stat_fn, n_boot, rng)
            for metric, (est, lo, hi) in res.items():
                rows.append({'scope': scope, 'class': name, 'n_cases': len(sub),
                             'n_patients': n_pat, 'metric': metric,
                             'estimate': est, 'ci_low': lo, 'ci_high': hi})
    return pd.DataFrame(rows)


def format_table(long):
    """One formatted cell 'est [lo, hi]' per scope/class/metric."""
    def fmt(r):
        return f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"
    long = long.copy()
    long['cell'] = long.apply(fmt, axis=1)
    table = long.pivot_table(index=['scope', 'class', 'n_cases', 'n_patients'],
                             columns='metric', values='cell', aggfunc='first')
    order = [m for m in ['dice', 'nsd', 'hd95_mm', 'icc', 'pearson', 'bias_ml', 'loa_low_ml',
                         'loa_high_ml', 'mae_ml', 'bias_pct', 'mape_pct'] if m in table.columns]
    scope_order = {s: i for i, s in enumerate(sorted(long['scope'].unique(), key=lambda s: (s == 'pooled', s)))}
    table = table[order].reset_index()
    table['_o'] = table['scope'].map(scope_order)
    return table.sort_values(['_o', 'class']).drop(columns='_o')


# ----------------------------------------------------------------------------
# Population tables
# ----------------------------------------------------------------------------

def cell(d):
    """'participants (examinations)' for a subset of cases."""
    return f"{d['patient'].nunique()} ({len(d)})" if len(d) else '–'


def population_tables(df):
    """Participants (examinations) per site and per scanner, split by cohort."""
    df = df.assign(cohort=df['patient'].map(cohort_of))
    tables = {}
    for name, by in [('site', 'group'), ('scanner', 'scanner')]:
        rows = {}
        for key, g in df.groupby(by):
            rows[key] = {'Patients': cell(g[g['cohort'] == 'Patients']),
                         'Controls': cell(g[g['cohort'] == 'Controls']),
                         'Total': cell(g)}
        rows['Total'] = {'Patients': cell(df[df['cohort'] == 'Patients']),
                         'Controls': cell(df[df['cohort'] == 'Controls']),
                         'Total': cell(df)}
        tables[name] = pd.DataFrame(rows).T
    # Participants per fold and scanner, to check that the folds are balanced
    tables['scanner_by_fold'] = (df.groupby(['fold', 'scanner'])['patient'].nunique()
                                 .unstack('scanner').fillna(0).astype(int))
    return tables


# ----------------------------------------------------------------------------
# Quality checks and worst cases
# ----------------------------------------------------------------------------

def quality_table(df, min_ref_ml):
    """Counts that belong in the Methods/Results: empty masks, failures, small volumes."""
    rows = []
    for name in CLASS_MAP.values():
        dice = df[f'dice_{name}']
        ref_empty = df[f'n_ref_{name}'] == 0
        pred_empty = df[f'n_pred_{name}'] == 0
        rows.append({
            'class': name,
            'n_cases': len(df),
            'reference_empty': int(ref_empty.sum()),
            'prediction_empty': int(pred_empty.sum()),
            'both_empty (Dice undefined, not in mean)': int((ref_empty & pred_empty).sum()),
            'n_cases_in_mean_dice': int(dice.notna().sum()),
            'dice_below_0.7': int((dice < 0.7).sum()),
            'dice_below_0.5': int((dice < 0.5).sum()),
            f'reference_below_{min_ref_ml:g}_mL (not in % metrics)': int((df[f'vol_ref_ml_{name}'] < min_ref_ml).sum()),
        })
    return pd.DataFrame(rows)


def worst_cases(df, n_worst, min_ref_ml, with_surface=False):
    """The n_worst cases with the lowest Dice per class: all cases, and cases with a
    reference volume of at least min_ref_ml (real failures instead of tiny structures).
    With surface metrics also the cases with the highest HD95."""
    parts = []
    for name in CLASS_MAP.values():
        cols = ['case', 'fold', 'scanner', 'group', f'dice_{name}']
        names = ['case', 'fold', 'scanner', 'site', 'dice']
        if with_surface:
            cols += [f'nsd_{name}', f'hd95_mm_{name}']
            names += ['nsd', 'hd95_mm']
        cols += [f'vol_ref_ml_{name}', f'vol_pred_ml_{name}']
        names += ['vol_ref_ml', 'vol_pred_ml']
        big = df[df[f'vol_ref_ml_{name}'] >= min_ref_ml]
        lists = [('lowest Dice', df.nsmallest(n_worst, f'dice_{name}')),
                 (f'lowest Dice, reference >= {min_ref_ml:g} mL', big.nsmallest(n_worst, f'dice_{name}'))]
        if with_surface:
            lists.append(('highest HD95', df.nlargest(n_worst, f'hd95_mm_{name}')))
        for label, sub in lists:
            w = sub[cols].copy()
            w.columns = names
            w.insert(0, 'list', label)
            w.insert(0, 'class', name)
            parts.append(w)
    return pd.concat(parts, ignore_index=True)


def descriptive_table(df, with_surface):
    """Mean, SD, median and interquartile range per scope and class (no bootstrap).

    Dice, and especially HD95, are skewed: the median (IQR) is often the better summary.
    """
    metrics = ['dice'] + (['nsd', 'hd95_mm'] if with_surface else [])
    scopes = [(f'fold_{f}', df[df['fold'] == f]) for f in sorted(df['fold'].unique())]
    scopes += [(f'scanner: {s}', df[df['scanner'] == s]) for s in sorted(df['scanner'].unique())]
    scopes.append(('pooled', df))
    rows = []
    for scope, sub in scopes:
        for name in CLASS_MAP.values():
            for metric in metrics:
                x = sub[f'{metric}_{name}'].dropna()
                rows.append({'scope': scope, 'class': name, 'metric': metric, 'n': len(x),
                             'mean': x.mean(), 'sd': x.std(ddof=1), 'median': x.median(),
                             'q25': x.quantile(0.25), 'q75': x.quantile(0.75),
                             'min': x.min(), 'max': x.max()})
    return pd.DataFrame(rows)


def add_median_columns(table, desc):
    """Add 'median (IQR)' columns to the formatted results table."""
    d = desc.copy()
    d['cell'] = d.apply(lambda r: f"{r['median']:.3f} ({r['q25']:.3f}-{r['q75']:.3f})", axis=1)
    wide = d.pivot_table(index=['scope', 'class'], columns='metric', values='cell', aggfunc='first')
    wide.columns = [f'{c}_median_iqr' for c in wide.columns]
    return table.merge(wide.reset_index(), on=['scope', 'class'], how='left')


def spacing_table(df):
    """Voxel sizes (mm) per scanner group, with the number of cases and participants."""
    s = df.assign(spacing=df.apply(
        lambda r: f"{r['spacing_x']:.2f} x {r['spacing_y']:.2f} x {r['spacing_z']:.2f}", axis=1))
    return (s.groupby(['scanner', 'spacing'])
             .agg(n_cases=('case', 'size'), n_participants=('patient', 'nunique'))
             .reset_index())


def check_tolerance(df, tolerance):
    """Warn if the NSD tolerance is smaller than the in-plane voxel size of some scanner groups."""
    coarse = df.groupby('scanner')[['spacing_x', 'spacing_y']].max().max(axis=1)
    coarse = coarse[coarse > tolerance]
    if len(coarse):
        msg = (f'NSD tolerance of {tolerance:g} mm is smaller than the in-plane voxel size of: '
               + ', '.join(f'{k} ({v:.2f} mm)' for k, v in coarse.items())
               + '. NSD is then hard to compare between scanner groups; consider a tolerance >= voxel size.')
        print('WARNING:', msg)
        logging.warning(msg)


# ----------------------------------------------------------------------------
# Excel output
# ----------------------------------------------------------------------------

def write_excel(path, sheets):
    """Write the tables to one Excel file, one sheet per table.

    sheets: {sheet name: (dataframe, write_index)}. Needs openpyxl; if it is not
    installed the Excel file is skipped and the csv files remain.
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        logging.warning('openpyxl not installed: Excel file skipped')
        print('openpyxl not installed, Excel file skipped (install with: pip install openpyxl)')
        return
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, (frame, with_index) in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=with_index)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = 'A2'
            for column in ws.columns:  # widen the columns to fit their content
                width = max(len(str(c.value)) if c.value is not None else 0 for c in column)
                ws.column_dimensions[column[0].column_letter].width = min(width + 2, 60)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def boxplot(df, by, out_png, title):
    keys = sorted(df[by].unique(), key=str)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.2 * len(keys) + 4), 4.5), sharey=True)
    for ax, name in zip(axes, CLASS_MAP.values()):
        data = [df.loc[df[by] == k, f'dice_{name}'].dropna() for k in keys]
        ax.boxplot(data, showfliers=True)
        # Set the tick labels afterwards: the 'labels' keyword differs per matplotlib version
        ax.set_xticks(range(1, len(keys) + 1))
        ax.set_xticklabels([str(k) for k in keys], rotation=20, ha='right')
        ax.set_title(f'{title}: {name}')
        ax.set_xlabel(by)
        ax.set_ylim(0, 1.02)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Dice')
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def volume_figures(df, out_abs, out_rel, min_ref_ml=MIN_REF_ML):
    """Scatter + Bland-Altman in mL, and Bland-Altman on percentage differences."""
    folds = sorted(df['fold'].unique())
    colors = {f: plt.cm.tab10(i) for i, f in enumerate(folds)}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    fig_rel, axes_rel = plt.subplots(1, 2, figsize=(10, 4.5))
    for col, name in enumerate(CLASS_MAP.values()):
        ref, pred = df[f'vol_ref_ml_{name}'], df[f'vol_pred_ml_{name}']
        ok = ref.notna() & pred.notna()
        d = df[ok]
        ref, pred = d[f'vol_ref_ml_{name}'], d[f'vol_pred_ml_{name}']
        c = [colors[f] for f in d['fold']]

        ax = axes[0, col]
        ax.scatter(ref, pred, s=12, alpha=0.6, c=c)
        top = max(ref.max(), pred.max()) * 1.05
        ax.plot([0, top], [0, top], 'k--', lw=1)
        stats = agreement_stats(ref.to_numpy(), pred.to_numpy(), min_ref_ml)
        ax.set_title(f'{name}: ICC {stats["icc"]:.3f}, r {stats["pearson"]:.3f}')
        ax.set_xlabel('Reference volume (mL)')
        ax.set_ylabel('Predicted volume (mL)')

        ax = axes[1, col]
        mean, diff = (ref + pred) / 2, pred - ref
        ax.scatter(mean, diff, s=12, alpha=0.6, c=c)
        for y, ls in [(stats['bias_ml'], '-'), (stats['loa_low_ml'], '--'), (stats['loa_high_ml'], '--')]:
            ax.axhline(y, color='k', ls=ls, lw=1)
        ax.set_title(f'Bland-Altman: bias {stats["bias_ml"]:.2f} mL, '
                     f'limits {stats["loa_low_ml"]:.2f} to {stats["loa_high_ml"]:.2f}')
        ax.set_xlabel('Mean volume (mL)')
        ax.set_ylabel('Predicted - reference (mL)')

        # Relative: difference as a percentage of the mean volume
        ax = axes_rel[col]
        nz = ref >= min_ref_ml  # tiny reference volumes give meaningless percentages
        pct = 100 * diff[nz] / mean[nz]
        ax.scatter(mean[nz], pct, s=12, alpha=0.6, c=[cc for cc, k in zip(c, nz) if k])
        b, sd = pct.mean(), pct.std(ddof=1)
        for y, ls in [(b, '-'), (b - 1.96 * sd, '--'), (b + 1.96 * sd, '--')]:
            ax.axhline(y, color='k', ls=ls, lw=1)
        ax.set_title(f'{name}: bias {b:.1f}%, limits {b - 1.96 * sd:.1f} to {b + 1.96 * sd:.1f}%\n'
                     f'(reference >= {min_ref_ml:g} mL, n = {int(nz.sum())})', fontsize=10)
        ax.set_xlabel('Mean volume (mL)')
        ax.set_ylabel('Difference (% of mean)')

    handles = [plt.Line2D([], [], marker='o', ls='', color=colors[f], label=f'fold {f}') for f in folds]
    axes[0, 0].legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_abs, dpi=200)
    fig_rel.tight_layout()
    fig_rel.savefig(out_rel, dpi=200)
    plt.close(fig)
    plt.close(fig_rel)


# ----------------------------------------------------------------------------

def run(build, surface=False, tolerance=1.0, n_boot=2000, seed=42,
        min_ref_ml=MIN_REF_ML, n_worst=10):

    build_rsf = os.path.join(build, PIPELINE)
    train_dir = os.path.join(build_rsf, 'stage_10_train', DATASET, TRAINER)
    labels_dir = os.path.join(build_rsf, 'stage_8_build_training_data', DATASET, 'labelsTr')
    crossval_dir = os.path.join(train_dir, CROSSVAL)
    out = os.path.join(build_rsf, 'results_summary')
    os.makedirs(out, exist_ok=True)

    # Dice per case, from nnU-Net's own validation summaries
    rows = []
    for fold in range(N_FOLDS):
        rows += read_fold(train_dir, fold)
    df = pd.DataFrame(rows)
    print(f'{len(df)} cases, {df["patient"].nunique()} patients')

    # Voxel sizes (mm) and volumes in mL
    spacing = {c: voxel_spacing(labels_dir, c) for c in df['case']}
    for i, axis in enumerate('xyz'):
        df[f'spacing_{axis}'] = df['case'].map(lambda c, i=i: spacing[c][i])
    vox = {c: float(np.prod(s)) / 1000.0 for c, s in spacing.items()}
    for name in CLASS_MAP.values():
        df[f'vol_ref_ml_{name}'] = df['n_ref_' + name] * df['case'].map(vox)
        df[f'vol_pred_ml_{name}'] = df['n_pred_' + name] * df['case'].map(vox)

    if surface:
        surf = compute_surface(list(df['case']), labels_dir, crossval_dir, tolerance)
        surf.to_csv(os.path.join(out, 'surface_per_case.csv'), index=False)
        df = df.merge(surf, on='case', how='left')
        check_tolerance(df, tolerance)

    df.to_csv(os.path.join(out, 'dice_per_case.csv'), index=False)

    # Participants (examinations) per site and scanner, for Table 1
    population = population_tables(df)
    for name, t in population.items():
        t.to_csv(os.path.join(out, f'population_{name}.csv'))
        print(f'\nPopulation per {name.replace("_", " ")}:')
        print(t.to_string())
    print()

    # Empty masks, failures and small volumes; and the worst cases for the failure analysis
    quality = quality_table(df, min_ref_ml)
    worst = worst_cases(df, n_worst, min_ref_ml, with_surface=surface)
    spacing_per_scanner = spacing_table(df)
    spacing_per_scanner.to_csv(os.path.join(out, 'voxel_spacing.csv'), index=False)
    print('Voxel size per scanner group:')
    print(spacing_per_scanner.to_string(index=False))
    print()
    quality.to_csv(os.path.join(out, 'data_quality.csv'), index=False)
    worst.to_csv(os.path.join(out, 'worst_cases.csv'), index=False)
    print('Data quality:')
    print(quality.T.to_string(header=False))
    print()

    long = summarise(df, n_boot, seed, with_surface=surface, min_ref_ml=min_ref_ml)
    long.to_csv(os.path.join(out, 'results_long.csv'), index=False)
    desc = descriptive_table(df, surface)
    desc.to_csv(os.path.join(out, 'descriptives.csv'), index=False)
    table = add_median_columns(format_table(long), desc)
    table.to_csv(os.path.join(out, 'results_table.csv'), index=False)

    # Everything in one Excel file, one sheet per table
    write_excel(os.path.join(out, 'results_summary.xlsx'), {
        'Population_site': (population['site'], True),
        'Population_scanner': (population['scanner'], True),
        'Scanner_by_fold': (population['scanner_by_fold'], True),
        'Results': (table, False),
        'Results_long': (long, False),
        'Descriptives': (desc, False),
        'Voxel_spacing': (spacing_per_scanner, False),
        'Data_quality': (quality, False),
        'Worst_cases': (worst, False),
        'Per_case': (df, False),
    })

    boxplot(df, 'fold', os.path.join(out, 'dice_by_fold.png'), 'Dice per fold')
    boxplot(df, 'group', os.path.join(out, 'dice_by_group.png'), 'Dice per site')
    boxplot(df, 'scanner', os.path.join(out, 'dice_by_scanner.png'), 'Dice per scanner')
    volume_figures(df, os.path.join(out, 'volume_agreement.png'),
                   os.path.join(out, 'volume_agreement_rel.png'), min_ref_ml)

    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(table.to_string(index=False))
    print(f'Written to {out}')


if __name__ == '__main__':

    BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    parser.add_argument("--surface", action="store_true",
                        help="Also compute HD95 and normalized surface Dice (slow)")
    parser.add_argument("--tolerance", type=float, default=2.0,
                        help="Tolerance in mm for the normalized surface Dice")
    parser.add_argument("--boot", type=int, default=2000,
                        help="Number of patient-level bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the bootstrap")
    parser.add_argument("--min-ref-ml", type=float, default=MIN_REF_ML,
                        help="Percentage errors only use cases with at least this reference volume (mL)")
    parser.add_argument("--n-worst", type=int, default=10,
                        help="Number of lowest-Dice cases to list per class")
    args = parser.parse_args()

    run(args.build, surface=args.surface, tolerance=args.tolerance,
        n_boot=args.boot, seed=args.seed, min_ref_ml=args.min_ref_ml, n_worst=args.n_worst)