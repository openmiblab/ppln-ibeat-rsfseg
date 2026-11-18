"""P-values for the statements in the results text, from results_summary.xlsx (sheet Per_case).

Tests
  1. Predicted versus reference volume (per class): paired Wilcoxon signed-rank test, plus the
     paired t-test and the mean / median difference.
  2. Dice, volunteers versus patients (per class): Mann-Whitney U test (two-sided).
  3. Reference volume, volunteers versus patients (per class): Mann-Whitney U test (two-sided).

Examinations of the same person are not independent. With --subject-col the values are first
averaged per subject (one value per person) and the tests are done on those, which is the
safe choice when a person has more than one examination. Without it every examination counts as
one observation.

Volunteers are recognised by their case name (default: contains C01, C02, ...; see
--volunteer-regex). Alternatively give --cohort-col and --volunteer-values.

Usage:
    python stats_tests.py --xlsx <path>/results_summary.xlsx
    python stats_tests.py --xlsx ... --subject-col patient
    python stats_tests.py --xlsx ... --cohort-col group2 --volunteer-values Volunteer
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats

CLASSES = ['left', 'right']


def fmt_p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def find_cohort(df, col, vol_values):
    if col:
        if col not in df.columns:
            raise SystemExit(f'Column {col} not found. Columns: {list(df.columns)}')
        if not vol_values:
            raise SystemExit(f'Give --volunteer-values; values in {col}: {sorted(df[col].dropna().astype(str).unique())}')
        return col, set(vol_values)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            vals = df[c].dropna().astype(str).unique()
            hit = {v for v in vals if 'volunteer' in v.lower() or 'healthy' in v.lower()}
            if hit:
                return c, hit
    raise SystemExit('Could not find the volunteer/patient column. Use --cohort-col and --volunteer-values.\n'
                     f'Columns: {list(df.columns)}')


def per_subject(d, subject_col, cols):
    """One row per subject: mean of the numeric columns (cohort is constant within a subject)."""
    agg = {c: 'mean' for c in cols}
    agg['is_vol'] = 'first'
    return d.groupby(subject_col, as_index=False).agg(agg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xlsx', required=True)
    p.add_argument('--subject-col', default=None, help='Column with a subject id: average per subject first')
    p.add_argument('--volunteer-regex', default=r'(?<![A-Za-z])C\d{2}',
                   help='Volunteers are the cases whose name matches this (default: C01, C02, ...)')
    p.add_argument('--subject-regex', default=None,
                   help='Regex with one capture group that extracts the subject id from the case name, '
                        r'e.g. "^([^_]+_[^_]+)_" ; the tests are then done on subject averages')
    p.add_argument('--cohort-col', default=None)
    p.add_argument('--volunteer-values', nargs='+', default=None)
    args = p.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name='Per_case')
    if args.cohort_col:
        ccol, vvals = find_cohort(df, args.cohort_col, args.volunteer_values)
        df['is_vol'] = df[ccol].astype(str).isin(vvals)
        print(f'Cohort column: {ccol}; volunteer values: {sorted(vvals)}')
    else:
        df['is_vol'] = df['case'].astype(str).str.contains(args.volunteer_regex, regex=True)
        print(f'Volunteers = case name matching {args.volunteer_regex!r}')
        print('  e.g. volunteers:', list(df.loc[df.is_vol, 'case'].head(3)),
              '| patients:', list(df.loc[~df.is_vol, 'case'].head(3)))
    print(f'  volunteers {int(df.is_vol.sum())} examinations, patients {int((~df.is_vol).sum())}')
    if df.is_vol.sum() == 0 or (~df.is_vol).sum() == 0:
        raise SystemExit('One of the groups is empty: check --volunteer-regex against the case names above.')
    if args.subject_regex:
        ext = df['case'].astype(str).str.extract(args.subject_regex)[0]
        if ext.isna().any():
            raise SystemExit(f'--subject-regex did not match for e.g. {list(df.loc[ext.isna(), "case"].head(3))}')
        df['_subject'] = ext
        args.subject_col = '_subject'
    if args.subject_col:
        if args.subject_col not in df.columns:
            raise SystemExit(f'Column {args.subject_col} not found. Columns: {list(df.columns)}')
        print(f'Values averaged per subject ({args.subject_col}); n below = subjects.')
    else:
        print('Every examination counted as one observation (use --subject-col if persons have several).')

    num = [f'{m}_{c}' for c in CLASSES for m in ('dice', 'vol_ref_ml', 'vol_pred_ml')]
    for c in CLASSES:
        df[f'vdiff_{c}'] = df[f'vol_pred_ml_{c}'] - df[f'vol_ref_ml_{c}']
    num += [f'vdiff_{c}' for c in CLASSES]

    d = per_subject(df, args.subject_col, num) if args.subject_col else df

    print('\n1. Predicted minus reference volume (paired)')
    for c in CLASSES:
        x = d[f'vdiff_{c}'].dropna().to_numpy(float)
        w = stats.wilcoxon(x)
        t = stats.ttest_1samp(x, 0.0)
        print(f'  {c}: n={len(x)}, mean diff {x.mean():.3f} mL, median diff {np.median(x):.3f} mL, '
              f'Wilcoxon p {fmt_p(w.pvalue)}, paired t p {fmt_p(t.pvalue)}')

    for title, metric, unit in [('2. Dice, volunteers vs patients', 'dice', ''),
                                ('3. Reference volume, volunteers vs patients', 'vol_ref_ml', ' mL')]:
        print(f'\n{title}')
        for c in CLASSES:
            col = f'{metric}_{c}'
            v = d.loc[d.is_vol, col].dropna().to_numpy(float)
            q = d.loc[~d.is_vol, col].dropna().to_numpy(float)
            u = stats.mannwhitneyu(v, q, alternative='two-sided')
            print(f'  {c}: volunteers n={len(v)} median {np.median(v):.3f}{unit} (mean {v.mean():.3f}), '
                  f'patients n={len(q)} median {np.median(q):.3f}{unit} (mean {q.mean():.3f}), '
                  f'Mann-Whitney p {fmt_p(u.pvalue)}')


if __name__ == '__main__':
    main()