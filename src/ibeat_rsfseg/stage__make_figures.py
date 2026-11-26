"""Figures for the RSF segmentation paper, made from results_summary.xlsx.

Run results_summary.py first (with --surface to get NSD and HD95); this script reads
its Excel file and writes the figures as png and pdf.

Figures (in <folder of the xlsx>/figures):
    fig_metrics_by_scanner   Dice (and NSD, HD95 if present) per case: all cases and
                             per scanner group, left and right class. Box = median and
                             IQR, whiskers 1.5 x IQR, dots = individual examinations.
                             HD95 is drawn on a log axis (values below 0.1 mm are drawn at 0.1).
    fig_metrics_by_site      The same, per site (Site 1-6).
    fig_metrics_by_fold      The same, per cross-validation fold.
    fig_volume_agreement     Per class: predicted versus reference volume, Bland-Altman in mL
                             (bias and limits of agreement with their 95% CI from the patient
                             bootstrap) and Bland-Altman in % of the mean volume.
    fig_examples             (--examples) Example segmentations: median, lower quartile,
                             poor and largest-HD95 case, with reference and prediction contours.
                             The crop is tight around the masks in the shown slice (see --margin),
                             and for both kidneys the slice is chosen where left and right are
                             both well visible.
                             Needs the build folder (images, labels and cross-validation
                             predictions). Shown in radiological orientation (anterior up, patient right on the left of the image),
                             reoriented using the nifti affine.

Usage:
    python make_figures.py --xlsx <path>/results_summary.xlsx
    python make_figures.py --xlsx <path>/results_summary.xlsx --build <iBEAt_Build> --examples
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['left', 'right']
SCANNER_ORDER = ['Siemens', 'Philips', 'Philips (old hybrid)', 'GE PET-MR']
# Okabe-Ito colour-blind safe palette
COLORS = {'Siemens': '#0072B2', 'Philips': '#E69F00',
          'Philips (old hybrid)': '#CC79A7', 'GE PET-MR': '#009E73'}
# Numbers used in the figures instead of site names / scanner models.
# Check these against Tables 2 and 3 of the manuscript and edit if needed.
SITE_NUMBER = {'Bari': 1, 'Bordeaux': 2, 'Exeter': 3, 'Leeds': 4, 'Turku': 5, 'Sheffield': 6}
SCANNER_NUMBER = {'GE PET-MR': 1, 'Philips': 2, 'Philips (old hybrid)': 3, 'Siemens': 4}
GREY = '#7f7f7f'
CLASS_COLOR = {'left': '#D55E00', 'right': '#0072B2'}   # contours in the example figure

PIPELINE = 'rsfseg'
DATASET = 'Dataset001_iBEAtRSF'
TRAINER = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
CROSSVAL = 'crossval_results_folds_0_1_2_3_4'
FAT_CHANNEL = 3   # channel of the fat image in imagesTr (out_phase, in_phase, water, fat, FF)

plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'axes.titlesize': 15,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def save(fig, base, formats, dpi):
    for fmt in formats:
        fig.savefig(f'{base}.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('Saved', base)


def panel_letter(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=16, fontweight='bold')


# ----------------------------------------------------------------------------
# Figure 1: distribution of Dice, NSD and HD95
# ----------------------------------------------------------------------------

def metrics_figure(df, group_col, groups, colors, base, formats, dpi, rotate=0,
                   include_all=False, xlabel=None, show_n=True):
    """Classic box plots (orange median, open circles = outliers beyond 1.5 x IQR)."""
    metrics = [('dice', 'Dice', False)]
    if 'nsd_left' in df.columns:
        metrics.append(('nsd', 'NSD', False))
    if 'hd95_mm_left' in df.columns:
        metrics.append(('hd95_mm', 'HD95 (mm)', True))
    labels = (['All'] if include_all else []) + list(groups)

    fig, axes = plt.subplots(len(metrics), 2, squeeze=False, sharey='row',
                             figsize=(2 * (1.15 * len(labels) + 1.8), 4.2 * len(metrics)))
    for r, (metric, ylabel, log) in enumerate(metrics):
        for c, cls in enumerate(CLASSES):
            ax = axes[r, c]
            col = f'{metric}_{cls}'
            data, ticks = [], []
            for g in labels:
                sub = df if g == 'All' else df[df[group_col] == g]
                x = sub[col].dropna().to_numpy(float)
                if log:
                    x = np.maximum(x, 0.1)
                data.append(x)
                ticks.append(f'{g}\n(n={len(x)})' if show_n else str(g))
            pos = np.arange(1, len(labels) + 1)
            keep = [(p, x) for p, x in zip(pos, data) if len(x)]
            ax.boxplot([x for _, x in keep], positions=[p for p, _ in keep], widths=0.6,
                       flierprops=dict(marker='o', markerfacecolor='none', markersize=5))
            ax.set_xticks(pos)
            ax.set_xticklabels(ticks, fontsize=13, rotation=rotate, ha='right' if rotate else 'center')
            if xlabel:
                ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(ylabel)
            if log:
                ax.set_yscale('log')
            else:
                ax.set_ylim(0, 1.02)
            ax.grid(axis='y', alpha=0.3, linewidth=0.6)
            ax.set_title(f'{ylabel} per {xlabel or group_col}: {cls}' if xlabel else f'{ylabel}: {cls}',
                         fontsize=16)
    fig.tight_layout()
    save(fig, base, formats, dpi)


# ----------------------------------------------------------------------------
# Figure 2: volume agreement
# ----------------------------------------------------------------------------

def pooled_stat(long, cls, metric):
    """(estimate, ci_low, ci_high) of a pooled statistic from Results_long, or None."""
    if long is None:
        return None
    r = long[(long['scope'] == 'pooled') & (long['class'] == cls) & (long['metric'] == metric)]
    if r.empty:
        return None
    return float(r.estimate.iloc[0]), float(r.ci_low.iloc[0]), float(r.ci_high.iloc[0])


def ba_line(ax, est, lo, hi, ls):
    ax.axhline(est, color='black', lw=1.0, ls=ls)
    if lo is not None and np.isfinite(lo) and np.isfinite(hi):
        ax.axhspan(lo, hi, color='black', alpha=0.10, lw=0)


def volume_figure(df, long, min_ref_ml, colors, base, formats, dpi):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for r, cls in enumerate(CLASSES):
        ref = df[f'vol_ref_ml_{cls}'].to_numpy(float)
        pred = df[f'vol_pred_ml_{cls}'].to_numpy(float)
        ok = np.isfinite(ref) & np.isfinite(pred)
        scanner = df['scanner'].to_numpy()[ok]
        ref, pred = ref[ok], pred[ok]
        pcol = [colors.get(s, GREY) for s in scanner]
        diff, mean = pred - ref, (pred + ref) / 2

        # (a) predicted versus reference
        ax = axes[r, 0]
        ax.scatter(ref, pred, s=9, alpha=0.6, c=pcol, linewidths=0, rasterized=True)
        top = max(ref.max(), pred.max()) * 1.05
        ax.plot([0, top], [0, top], color='black', lw=0.8, ls='--')
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect('equal')
        ax.set_xlabel('Reference volume (mL)')
        ax.set_ylabel('Predicted volume (mL)')
        icc, pear = pooled_stat(long, cls, 'icc'), pooled_stat(long, cls, 'pearson')
        text = [f'n = {len(ref)}']
        if icc:
            text.insert(0, f'ICC = {icc[0]:.3f} ({icc[1]:.3f}-{icc[2]:.3f})')
        if pear:
            text.insert(1 if icc else 0, f'r = {pear[0]:.3f} ({pear[1]:.3f}-{pear[2]:.3f})')
        else:
            text.insert(0, f'r = {np.corrcoef(ref, pred)[0, 1]:.3f}')
        ax.text(0.04, 0.96, '\n'.join(text), transform=ax.transAxes, va='top', fontsize=12.5)
        ax.set_title(f'{cls.capitalize()} renal sinus fat', fontsize=15)
        panel_letter(ax, chr(97 + r * 3))

        # (b) Bland-Altman in mL, with the bootstrap CI of the bias and the limits
        ax = axes[r, 1]
        ax.scatter(mean, diff, s=9, alpha=0.6, c=pcol, linewidths=0, rasterized=True)
        sd = diff.std(ddof=1)
        lines = []
        for metric, fallback, ls in [('bias_ml', diff.mean(), '-'),
                                     ('loa_low_ml', diff.mean() - 1.96 * sd, '--'),
                                     ('loa_high_ml', diff.mean() + 1.96 * sd, '--')]:
            s = pooled_stat(long, cls, metric)
            est, lo, hi = s if s else (fallback, None, None)
            ba_line(ax, est, lo, hi, ls)
            lines.append(est)
        ax.set_title(f'bias {lines[0]:.2f} mL, limits {lines[1]:.2f} to {lines[2]:.2f}\n'
                     '(shading: 95% CI)', fontsize=13)
        ax.set_xlabel('Mean volume (mL)')
        ax.set_ylabel('Predicted - reference (mL)')
        panel_letter(ax, chr(97 + r * 3 + 1))

        # (c) Bland-Altman on percentage differences, large enough reference volumes only
        ax = axes[r, 2]
        nz = ref >= min_ref_ml
        pct = 100 * diff[nz] / mean[nz]
        ax.scatter(mean[nz], pct, s=9, alpha=0.6, c=[c for c, k in zip(pcol, nz) if k],
                   linewidths=0, rasterized=True)
        b, sdp = pct.mean(), pct.std(ddof=1)
        for y, ls in [(b, '-'), (b - 1.96 * sdp, '--'), (b + 1.96 * sdp, '--')]:
            ax.axhline(y, color='black', lw=1.0, ls=ls)
        ax.set_title(f'bias {b:.1f}%, limits {b - 1.96 * sdp:.1f} to {b + 1.96 * sdp:.1f}%\n'
                     f'(reference >= {min_ref_ml:g} mL, n = {int(nz.sum())})', fontsize=13)
        ax.set_xlabel('Mean volume (mL)')
        ax.set_ylabel('Difference (% of mean)')
        panel_letter(ax, chr(97 + r * 3 + 2))

    present = [s for s in SCANNER_ORDER + sorted(set(df['scanner']) - set(SCANNER_ORDER))
               if s in set(df['scanner'])]
    handles = [plt.Line2D([], [], marker='o', ls='', color=colors.get(s, GREY),
                          label=f'Scanner {SCANNER_NUMBER[s]}' if s in SCANNER_NUMBER else s, markersize=5)
               for s in present]
    axes[0, 0].legend(handles=handles, loc='lower right', fontsize=12, frameon=False)
    fig.tight_layout()
    save(fig, base, formats, dpi)


# ----------------------------------------------------------------------------
# Figure 3: example segmentations
# ----------------------------------------------------------------------------

def pick_cases(df, min_ref_ml):
    """Median, lower quartile, poor (reference volumes >= min_ref_ml) and largest HD95."""
    d = df.assign(score=df[['dice_left', 'dice_right']].mean(axis=1, skipna=True))
    d = d[d['score'].notna()]
    picks = [('Median Dice', (d['score'] - d['score'].median()).abs().idxmin()),
             ('Lower quartile', (d['score'] - d['score'].quantile(0.25)).abs().idxmin())]
    big = d[(d['vol_ref_ml_left'] >= min_ref_ml) & (d['vol_ref_ml_right'] >= min_ref_ml)]
    if len(big):
        picks.append(('Poor overlap', big['score'].idxmin()))
    if 'hd95_mm_left' in d.columns:
        picks.append(('Largest HD95', d[['hd95_mm_left', 'hd95_mm_right']].max(axis=1).idxmax()))
    chosen, seen = [], set()
    for label, idx in picks:
        if d.loc[idx, 'case'] not in seen:
            seen.add(d.loc[idx, 'case'])
            chosen.append((label, d.loc[idx]))
    return chosen


def load_display(path, dtype=None):
    """Load a nifti, reorient it to RAS using its affine, and flip x so that array index x
    increases towards the patient's left. Shown with arr[:, :, z].T and origin='lower'
    this gives the radiological view: anterior up, patient right on the left of the image.
    Returns (array, voxel sizes)."""
    import nibabel as nib
    nii = nib.as_closest_canonical(nib.load(path))
    arr = np.asarray(nii.dataobj)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr[::-1], nii.header.get_zooms()


def examples_figure(df, build, cases, min_ref_ml, anonymise, base, formats, dpi, margin=10, width=10.0):
    """One row per case, one axial slice showing BOTH kidneys: fat image, reference, prediction.
    The slice is the one where the smaller of the left and right areas (reference + prediction
    together) is largest, so left and right are both well visible. The crop is the bounding box
    of the labelled voxels in that slice plus `margin` voxels."""
    raw = os.path.join(build, PIPELINE, 'stage_8_build_training_data', DATASET)
    pred_dir = os.path.join(build, PIPELINE, 'stage_10_train', DATASET, TRAINER, CROSSVAL)

    if cases:
        rows = []
        for spec in cases:
            c = spec.split(':')[0]     # old-style 'case:both:error' is still accepted, the suffix is ignored
            hit = df[df['case'] == c]
            if hit.empty:
                raise SystemExit(f'Case {c} not found in Per_case')
            rows.append(hit.iloc[0])
    else:
        rows = [row for _, row in pick_cases(df, min_ref_ml)]

    prepared = []
    for i, row in enumerate(rows):
        case = row['case']
        img, zooms = load_display(os.path.join(raw, 'imagesTr', f'{case}_000{FAT_CHANNEL}.nii.gz'), np.float32)
        ref = np.rint(load_display(os.path.join(raw, 'labelsTr', f'{case}.nii.gz'))[0]).astype(int)
        pred = np.rint(load_display(os.path.join(pred_dir, f'{case}.nii.gz'))[0]).astype(int)
        if not (img.shape == ref.shape == pred.shape):
            raise SystemExit(f'{case}: image {img.shape}, label {ref.shape} and prediction {pred.shape} differ in shape')

        # slice where both kidneys are best visible
        a_l = ((ref == 1) | (pred == 1)).sum(axis=(0, 1))
        a_r = ((ref == 2) | (pred == 2)).sum(axis=(0, 1))
        area = np.minimum(a_l, a_r) * 1000 + (a_l + a_r)     # tie-break on total area
        z = int(area.argmax())

        any_xy = (ref[:, :, z] > 0) | (pred[:, :, z] > 0)
        xs, ys = np.where(any_xy) if any_xy.any() else (np.array([img.shape[0] // 2]), np.array([img.shape[1] // 2]))
        x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, img.shape[0])
        y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, img.shape[1])
        crop = img[x0:x1, y0:y1, z].T
        vmax = max(np.percentile(crop, 99.5), 1e-6)

        text = f'({chr(97 + i)}) Dice L {row["dice_left"]:.2f}, R {row["dice_right"]:.2f}' + \
               ('' if anonymise else f'\n{case}')
        aspect = zooms[1] / zooms[0]
        prepared.append(dict(crop=crop, vmax=vmax, aspect=aspect, ref=ref, pred=pred, box=(x0, x1, y0, y1), z=z,
                             text=text, height=crop.shape[0] * aspect / crop.shape[1]))

    # Row heights follow the shape of each crop, so there is no empty space between rows
    col_w = width / 3.05
    heights = [col_w * p['height'] for p in prepared]
    fig, axes = plt.subplots(len(prepared), 3, figsize=(width, sum(heights) + 0.5 + 0.1 * len(prepared)),
                             squeeze=False, gridspec_kw=dict(height_ratios=heights, hspace=0.04, wspace=0.03))
    for i, d in enumerate(prepared):
        x0, x1, y0, y1 = d['box']
        for j, (title, mask) in enumerate([('Fat image', None), ('Reference', d['ref']), ('Prediction', d['pred'])]):
            ax = axes[i, j]
            ax.imshow(d['crop'], cmap='gray', vmin=0, vmax=d['vmax'], origin='lower', aspect=d['aspect'])
            if mask is not None:
                for k, cls in enumerate(CLASSES, start=1):
                    sub = (mask[x0:x1, y0:y1, d['z']] == k).T
                    if sub.any() and not sub.all():
                        ax.contour(sub.astype(float), levels=[0.5], colors=[CLASS_COLOR[cls]], linewidths=1.3)
            if j == 0:   # orientation: radiological view
                for xt, ha, lab in [(0.02, 'left', 'R'), (0.98, 'right', 'L')]:
                    ax.text(xt, 0.5, lab, transform=ax.transAxes, ha=ha, va='center', color='white',
                            fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=15)
        axes[i, 0].text(0.02, 0.97, d['text'], transform=axes[i, 0].transAxes, va='top', ha='left',
                        fontsize=12.5, color='white',
                        bbox=dict(facecolor='black', alpha=0.6, pad=2, lw=0))

    handles = [plt.Line2D([], [], color=CLASS_COLOR[c], lw=1.5, label=f'{c} renal sinus fat') for c in CLASSES]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=13)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.97, bottom=0.04)
    save(fig, base, formats, dpi)


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlsx', required=True, help='results_summary.xlsx made by results_summary.py')
    parser.add_argument('--out', default=None, help='Output folder (default: figures next to the xlsx)')
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'])
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--min-ref-ml', type=float, default=1.0,
                        help='Percentage Bland-Altman only for reference volumes >= this (mL)')
    parser.add_argument('--examples', action='store_true', help='Also make the example segmentation figure')
    parser.add_argument('--build', default=None, help='iBEAt_Build folder (needed for --examples)')
    parser.add_argument('--cases', nargs='+', default=None,
                        help='Case ids for the example figure, e.g. 7128_152_Baseline 7128_153_Baseline '
                             '(default: automatic selection)')
    parser.add_argument('--margin', type=int, default=10,
                        help='Margin (voxels) around the masks in the example crop; smaller = more zoomed in')
    parser.add_argument('--anonymise', action='store_true', help='Do not print case ids in the example figure')
    args = parser.parse_args()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.xlsx)), 'figures')
    os.makedirs(out, exist_ok=True)

    xls = pd.ExcelFile(args.xlsx)
    df = xls.parse('Per_case')
    long = xls.parse('Results_long') if 'Results_long' in xls.sheet_names else None

    # Scanners and sites are shown by number (see SCANNER_NUMBER / SITE_NUMBER)
    scanner_names = sorted(set(df['scanner']), key=lambda n: SCANNER_NUMBER.get(n, 99))
    df_s = df.assign(scanner_label=df['scanner'].map(lambda n: f'Scanner {SCANNER_NUMBER.get(n, n)}'))
    metrics_figure(df_s, 'scanner_label', [f'Scanner {SCANNER_NUMBER.get(n, n)}' for n in scanner_names], {},
                   os.path.join(out, 'fig_metrics_by_scanner'), args.formats, args.dpi,
                   rotate=0, xlabel='scanner')
    site_names = sorted(set(df['group']), key=lambda n: SITE_NUMBER.get(n, 99))
    df_g = df.assign(site_label=df['group'].map(lambda n: f'Site {SITE_NUMBER.get(n, n)}'))
    metrics_figure(df_g, 'site_label', [f'Site {SITE_NUMBER.get(n, n)}' for n in site_names], {},
                   os.path.join(out, 'fig_metrics_by_site'), args.formats, args.dpi,
                   rotate=0, xlabel='site')
    print('Sites:   ', {f'Site {SITE_NUMBER.get(n, n)}': n for n in site_names})
    print('Scanners:', {f'Scanner {SCANNER_NUMBER.get(n, n)}': n for n in scanner_names})
    folds = sorted(df['fold'].unique())
    df_f = df.assign(fold_label=df['fold'].astype(str))
    metrics_figure(df_f, 'fold_label', [str(f) for f in folds], {}, os.path.join(out, 'fig_metrics_by_fold'),
                   args.formats, args.dpi, include_all=False, xlabel='fold', show_n=False)
    volume_figure(df, long, args.min_ref_ml, COLORS, os.path.join(out, 'fig_volume_agreement'),
                  args.formats, args.dpi)

    if args.examples:
        if not args.build:
            raise SystemExit('--examples needs --build <iBEAt_Build folder>')
        examples_figure(df, args.build, args.cases, args.min_ref_ml, args.anonymise,
                        os.path.join(out, 'fig_examples'), args.formats, args.dpi, margin=args.margin)


if __name__ == '__main__':
    main()