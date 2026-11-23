"""Extra figures for the RSF segmentation paper: small and large reference volumes and Dice versus volume.

Put this file next to make_figures.py (it imports its constants and helpers) and run it on the
same results_summary.xlsx.

Figures (in <folder of the xlsx>/figures):
    fig_dice_vs_volume        Dice versus reference volume (log x axis), left and right class,
                              coloured by scanner. Dashed lines: reference volume of 1 and 2 mL
                              and Dice = 0.5.
    small_<side>_<case>       (--build) One figure per examination with a reference volume below
                              --max-ref-ml (default 2 mL) for the chosen side (default: right):
                              fat image, reference and prediction, in the coronal slice (--plane
                              axial for axial) where the structure is largest, with a tight crop
                              around it. Radiological view: patient right on the left of the image,
                              superior up (coronal) or anterior up (axial).
    large_<side>_<case>       (--build --min-ref-ml 40) One figure per examination with a reference
                              volume above --min-ref-ml, sorted from large to small volume. Whole
                              field of view, reference and prediction side by side, no marker.

Usage:
    python make_figures_small_volume.py --xlsx <path>/results_summary.xlsx
    python make_figures_small_volume.py --xlsx <path>/results_summary.xlsx --build <iBEAt_Build>
    python make_figures_small_volume.py --xlsx ... --build ... --side both --max-ref-ml 2 --anonymise
    python make_figures_small_volume.py --xlsx ... --build ... --side both --min-ref-ml 40 --anonymise
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import make_figures as mf

CLASS_ID = {'left': 1, 'right': 2}
# Background image of the case figures: (channel in imagesTr, panel title). Set by --background.
CHANNELS = {'fat': (3, 'Fat image'), 'ff': (4, 'Fat fraction'), 'water': (2, 'Water image')}
BG = list(CHANNELS['fat'])


# ----------------------------------------------------------------------------
# Dice versus reference volume
# ----------------------------------------------------------------------------

def spearman(x, y):
    """Spearman rank correlation without needing scipy."""
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def dice_vs_volume_figure(df, base, formats, dpi, small_ml=2.0, log=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for c, cls in enumerate(mf.CLASSES):
        ax = axes[c]
        ref = df[f'vol_ref_ml_{cls}'].to_numpy(float)
        dice = df[f'dice_{cls}'].to_numpy(float)
        ok = np.isfinite(ref) & np.isfinite(dice) & (ref > 0)
        scanner = df['scanner'].to_numpy()[ok]
        ref, dice = ref[ok], dice[ok]
        ax.scatter(ref, dice, s=14, alpha=0.7, c=[mf.COLORS.get(s, mf.GREY) for s in scanner],
                   linewidths=0, rasterized=True)
        if log:
            ax.set_xscale('log')
        else:
            ax.set_xlim(0, ref.max() * 1.05)
        ax.set_ylim(0, 1.02)
        for x in (1.0, small_ml):
            ax.axvline(x, color='black', lw=0.8, ls='--', alpha=0.6)
        ax.axhline(0.5, color='black', lw=0.8, ls=':', alpha=0.6)
        ax.set_xlabel('Reference volume (mL, log scale)' if log else 'Reference volume (mL)')
        if c == 0:
            ax.set_ylabel('Dice')
        ax.grid(alpha=0.3, linewidth=0.6)
        rho = spearman(ref, dice)
        ax.text(0.04, 0.04, f'n = {len(ref)}\nSpearman rho = {rho:.2f}\n'
                f'n < {small_ml:g} mL: {int((ref < small_ml).sum())}\n'
                f'Dice < 0.5: {int((dice < 0.5).sum())}',
                transform=ax.transAxes, va='bottom', fontsize=12)
        ax.set_title(f'{cls.capitalize()} renal sinus fat', fontsize=15)
        mf.panel_letter(ax, chr(97 + c))
        # numbers for the manuscript text
        low = dice < 0.5
        print(f'{cls}: n={len(ref)}, ref < {small_ml:g} mL: {int((ref < small_ml).sum())}, '
              f'Dice < 0.5: {int(low.sum())}'
              + (f' (max reference volume among them {ref[low].max():.2f} mL)' if low.any() else ''))

    present = [s for s in mf.SCANNER_ORDER + sorted(set(df['scanner']) - set(mf.SCANNER_ORDER))
               if s in set(df['scanner'])]
    handles = [plt.Line2D([], [], marker='o', ls='', color=mf.COLORS.get(s, mf.GREY), markersize=5,
                          label=f'Scanner {mf.SCANNER_NUMBER[s]}' if s in mf.SCANNER_NUMBER else s)
               for s in present]
    axes[1].legend(handles=handles, loc='lower right', frameon=False, fontsize=12)
    fig.tight_layout()
    mf.save(fig, base, formats, dpi)


# ----------------------------------------------------------------------------
# One figure per small-volume examination
# ----------------------------------------------------------------------------

def load_plane(path, plane, dtype=None):
    """Load like mf.load_display (RAS, x flipped: index x increases towards the patient's left,
    y towards anterior, z towards superior), then permute the axes so that the slicing axis is
    last. Shown with arr[:, :, s].T and origin='lower':
      coronal: horizontal = left-right (patient right on the left of the image), vertical = superior up
      axial:   horizontal = left-right, vertical = anterior up
    Voxel sizes are taken from the affine, so they are right after reorientation.
    Returns (array, (voxel size horizontal, voxel size vertical))."""
    import nibabel as nib
    nii = nib.as_closest_canonical(nib.load(path))
    arr = np.asarray(nii.dataobj)
    if dtype is not None:
        arr = arr.astype(dtype)
    arr = arr[::-1]
    zx, zy, zz = np.sqrt((nii.affine[:3, :3] ** 2).sum(axis=0))
    if plane == 'coronal':
        return np.transpose(arr, (0, 2, 1)), (zx, zz)
    return arr, (zx, zy)


def case_figure(row, cls, raw, pred_dir, margin, min_size, anonymise, base, formats, dpi, width=9.0,
                plane='coronal'):
    case = row['case']
    k = CLASS_ID[cls]
    img, zooms = load_plane(os.path.join(raw, 'imagesTr', f'{case}_000{BG[0]}.nii.gz'), plane, np.float32)
    ref = np.rint(load_plane(os.path.join(raw, 'labelsTr', f'{case}.nii.gz'), plane)[0]).astype(int)
    pred = np.rint(load_plane(os.path.join(pred_dir, f'{case}.nii.gz'), plane)[0]).astype(int)
    if not (img.shape == ref.shape == pred.shape):
        raise SystemExit(f'{case}: image {img.shape}, label {ref.shape} and prediction {pred.shape} differ in shape')

    # slice where this structure (reference + prediction together) is largest
    target = (ref == k) | (pred == k)
    area = target.sum(axis=(0, 1))
    if area.max() == 0:
        area = ((ref > 0) | (pred > 0)).sum(axis=(0, 1))
    z = int(area.argmax())

    m = target[:, :, z]
    if not m.any():
        m = (ref[:, :, z] > 0) | (pred[:, :, z] > 0)
    if m.any():
        xs, ys = np.where(m)
    else:
        xs, ys = np.array([img.shape[0] // 2]), np.array([img.shape[1] // 2])
    # crop: bounding box plus margin, but never smaller than min_size voxels
    def span(lo, hi, n):
        half = max((hi - lo + 1) / 2 + margin, min_size / 2)
        c = (lo + hi) / 2
        a, b = int(round(c - half)), int(round(c + half)) + 1
        return max(a, 0), min(b, n)
    x0, x1 = span(xs.min(), xs.max(), img.shape[0])
    y0, y1 = span(ys.min(), ys.max(), img.shape[1])

    crop = img[x0:x1, y0:y1, z].T
    vmax = max(np.percentile(crop, 99.5), 1e-6)
    aspect = zooms[1] / zooms[0]    # vertical voxel size / horizontal voxel size
    h = crop.shape[0] * aspect / crop.shape[1]

    parts = [f'{cls.capitalize()}: Dice {row[f"dice_{cls}"]:.2f}',
             f'ref {row[f"vol_ref_ml_{cls}"]:.2f} mL, pred {row[f"vol_pred_ml_{cls}"]:.2f} mL']
    if f'hd95_mm_{cls}' in row.index and np.isfinite(row[f'hd95_mm_{cls}']):
        parts[0] += f', HD95 {row[f"hd95_mm_{cls}"]:.1f} mm'
    if not anonymise:
        parts.append(case)

    col_w = width / 3.05
    fig, axes = plt.subplots(1, 3, figsize=(width, col_w * h + 0.9), squeeze=False,
                             gridspec_kw=dict(wspace=0.03))
    for j, (title, mask) in enumerate([(BG[1], None), ('Reference', ref), ('Prediction', pred)]):
        ax = axes[0, j]
        ax.imshow(crop, cmap='gray', vmin=0, vmax=vmax, origin='lower', aspect=aspect)
        if mask is not None:
            for kk, name in enumerate(mf.CLASSES, start=1):
                sub = (mask[x0:x1, y0:y1, z] == kk).T
                if sub.any() and not sub.all():
                    ax.contour(sub.astype(float), levels=[0.5], colors=[mf.CLASS_COLOR[name]], linewidths=1.3)
        if j == 0:
            for xt, ha, lab in [(0.02, 'left', 'R'), (0.98, 'right', 'L')]:
                ax.text(xt, 0.5, lab, transform=ax.transAxes, ha=ha, va='center', color='white',
                        fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=15)
    axes[0, 0].text(0.02, 0.97, '\n'.join(parts), transform=axes[0, 0].transAxes, va='top', ha='left',
                    fontsize=11, color='white', bbox=dict(facecolor='black', alpha=0.6, pad=2, lw=0))
    handles = [plt.Line2D([], [], color=mf.CLASS_COLOR[c], lw=1.5, label=f'{c} renal sinus fat') for c in mf.CLASSES]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.88, bottom=0.10)
    mf.save(fig, base, formats, dpi)


# ----------------------------------------------------------------------------

def hd95_vs_dice_figure(df, base, formats, dpi, thr=20.0, dice_hi=0.8):
    """HD95 (log axis) versus Dice, per class. Shows whether large HD95 values go with poor
    overlap or with good overlap (a separate fragment). Prints the counts used in the text."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for c, cls in enumerate(mf.CLASSES):
        ax = axes[c]
        hd = df[f'hd95_mm_{cls}'].to_numpy(float)
        dice = df[f'dice_{cls}'].to_numpy(float)
        ok = np.isfinite(hd) & np.isfinite(dice)
        scanner = df['scanner'].to_numpy()[ok]
        hd, dice = np.maximum(hd[ok], 0.1), dice[ok]
        ax.scatter(dice, hd, s=14, alpha=0.7, c=[mf.COLORS.get(s, mf.GREY) for s in scanner],
                   linewidths=0, rasterized=True)
        ax.set_yscale('log')
        ax.set_xlim(0, 1.02)
        ax.axhline(thr, color='black', lw=0.8, ls='--', alpha=0.6)
        ax.axvline(dice_hi, color='black', lw=0.8, ls=':', alpha=0.6)
        ax.set_xlabel('Dice')
        if c == 0:
            ax.set_ylabel('HD95 (mm, log scale)')
        ax.grid(alpha=0.3, linewidth=0.6)
        big = hd > thr
        n_big, n_good = int(big.sum()), int((big & (dice >= dice_hi)).sum())
        ax.text(0.04, 0.04, f'n = {len(hd)}\nHD95 > {thr:g} mm: {n_big} ({100 * n_big / len(hd):.1f}%)\n'
                f'  of which Dice >= {dice_hi:g}: {n_good}',
                transform=ax.transAxes, va='bottom', fontsize=12)
        ax.set_title(f'{cls.capitalize()} renal sinus fat', fontsize=15)
        mf.panel_letter(ax, chr(97 + c))
        print(f'{cls}: HD95 > {thr:g} mm in {n_big} of {len(hd)} ({100 * n_big / len(hd):.1f}%); '
              f'Dice >= {dice_hi:g} in {n_good} of them')
    present = [s for s in mf.SCANNER_ORDER + sorted(set(df['scanner']) - set(mf.SCANNER_ORDER))
               if s in set(df['scanner'])]
    handles = [plt.Line2D([], [], marker='o', ls='', color=mf.COLORS.get(s, mf.GREY), markersize=5,
                          label=f'Scanner {mf.SCANNER_NUMBER[s]}' if s in mf.SCANNER_NUMBER else s)
               for s in present]
    axes[0].legend(handles=handles, loc='upper right', frameon=False, fontsize=12)
    fig.tight_layout()
    mf.save(fig, base, formats, dpi)


# ----------------------------------------------------------------------------
# Large HD95: main structure and the far-away segment
# ----------------------------------------------------------------------------

def load_xyz(path, dtype=None):
    """RAS, x flipped (index x increases towards the patient's left); array axes (x, y, z).
    Returns (array, voxel sizes (zx, zy, zz) from the affine)."""
    import nibabel as nib
    nii = nib.as_closest_canonical(nib.load(path))
    arr = np.asarray(nii.dataobj)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr[::-1], tuple(float(v) for v in np.sqrt((nii.affine[:3, :3] ** 2).sum(axis=0)))


def plane_slice(arr, plane, s):
    """2D slice in display orientation (rows = vertical, columns = left-right)."""
    return (arr[:, s, :] if plane == 'coronal' else arr[:, :, s]).T


def plane_info(plane, zooms):
    zx, zy, zz = zooms
    if plane == 'coronal':
        return (0, 2), zz / zx       # axes to sum over, vertical/horizontal voxel size
    return (0, 1), zy / zx


def crop_box(m2d, margin, min_size):
    """(r0, r1, c0, c1) around a 2D mask, at least min_size voxels wide."""
    rs, cs = np.where(m2d)
    def span(lo, hi, n):
        half = max((hi - lo + 1) / 2 + margin, min_size / 2)
        ctr = (lo + hi) / 2
        return max(int(round(ctr - half)), 0), min(int(round(ctr + half)) + 1, n)
    r0, r1 = span(rs.min(), rs.max(), m2d.shape[0])
    c0, c1 = span(cs.min(), cs.max(), m2d.shape[1])
    return r0, r1, c0, c1


def far_voxels(ref_k, pred_k, zooms, thr):
    """Voxels of one mask that lie more than thr mm from the other mask."""
    from scipy import ndimage
    out = np.zeros(ref_k.shape, bool)
    if ref_k.any() and pred_k.any():
        d_to_ref = ndimage.distance_transform_edt(~ref_k, sampling=zooms)
        d_to_pred = ndimage.distance_transform_edt(~pred_k, sampling=zooms)
        out |= pred_k & (d_to_ref > thr)
        out |= ref_k & (d_to_pred > thr)
    return out


def overview_figure(row, cls, raw, pred_dir, thr, plane, anonymise, base, formats, dpi, panel_width=5.2):
    """Whole field of view, reference and prediction side by side. Every contour is drawn on the
    slice it belongs to (no projections):
      row 1: the slice where both kidneys are best visible;
      row 2: the slice with most voxels that lie more than thr mm from the other mask (the separate
             fragment), only if that is another slice than row 1.
    Yellow circles mark those parts, in the slice(s) where they are present."""
    from matplotlib.patches import Circle
    from scipy import ndimage
    case = row['case']
    k = CLASS_ID[cls]
    img, zooms = load_xyz(os.path.join(raw, 'imagesTr', f'{case}_000{BG[0]}.nii.gz'), np.float32)
    ref = np.rint(load_xyz(os.path.join(raw, 'labelsTr', f'{case}.nii.gz'))[0]).astype(int)
    pred = np.rint(load_xyz(os.path.join(pred_dir, f'{case}.nii.gz'))[0]).astype(int)
    if not (img.shape == ref.shape == pred.shape):
        raise SystemExit(f'{case}: image {img.shape}, label {ref.shape} and prediction {pred.shape} differ in shape')
    zx, zy, zz = zooms
    sum_axes, _ = plane_info(plane, zooms)
    zh, zv = (zx, zz) if plane == 'coronal' else (zx, zy)

    # slice where both kidneys are visible (as in fig_examples)
    a_l = ((ref == 1) | (pred == 1)).sum(axis=sum_axes)
    a_r = ((ref == 2) | (pred == 2)).sum(axis=sum_axes)
    s_main = int((np.minimum(a_l, a_r) * 1000 + a_l + a_r).argmax())

    far = np.zeros(ref.shape, bool)
    for t in (thr, thr / 2):
        far = far_voxels(ref == k, pred == k, zooms, t)
        if far.any():
            break
    lab, n = ndimage.label(far)
    comps = [int(c) + 1 for c in (np.argsort(ndimage.sum(far, lab, range(1, n + 1)))[::-1][:3] if n else [])]

    rows = [('Slice with both kidneys', s_main)]
    if far.any():
        s_far = int(far.sum(axis=sum_axes).argmax())
        if s_far != s_main:
            rows.append(('Slice with the separate fragment', s_far))

    bg0 = plane_slice(img, plane, s_main)
    nr, nc = bg0.shape
    xs, ys = (np.arange(nc) + 0.5) * zh, (np.arange(nr) + 0.5) * zv
    info = f'{cls.capitalize()}: Dice {row[f"dice_{cls}"]:.2f}'
    if np.isfinite(row.get(f'hd95_mm_{cls}', np.nan)):
        info += f', HD95 {row[f"hd95_mm_{cls}"]:.0f} mm'
    info += f'\nref {row[f"vol_ref_ml_{cls}"]:.1f} mL, pred {row[f"vol_pred_ml_{cls}"]:.1f} mL'
    if not anonymise:
        info += f'\n{case}'
    top_in, bot_in = 0.45 + 0.2 * (info.count('\n') + 1), 0.7
    img_h = panel_width * (nr * zv) / (nc * zh)
    H = len(rows) * img_h + top_in + bot_in + 0.1 * len(rows)

    fig, axes = plt.subplots(len(rows), 2, figsize=(2 * panel_width, H), squeeze=False)
    any_circle = False
    for i, (label, s) in enumerate(rows):
        bg = plane_slice(img, plane, s)
        vmax = max(np.percentile(bg, 99.5), 1e-6)
        for j, (title, mask) in enumerate([('Reference', ref), ('Prediction', pred)]):
            ax = axes[i, j]
            ax.imshow(bg, cmap='gray', vmin=0, vmax=vmax, origin='lower', extent=[0, nc * zh, 0, nr * zv],
                      aspect='equal')
            for kk, name in enumerate(mf.CLASSES, start=1):
                m2 = plane_slice(mask == kk, plane, s)
                if m2.any() and not m2.all():
                    ax.contour(xs, ys, m2.astype(float), levels=[0.5], colors=[mf.CLASS_COLOR[name]], linewidths=1.4)
            for comp in comps:
                c2 = plane_slice(lab == comp, plane, s)
                if c2.any():
                    rr, cc = np.where(c2)
                    rad = max(8.0, 0.5 * max(np.ptp(cc) * zh, np.ptp(rr) * zv) + 5.0)
                    ax.add_patch(Circle(((cc.mean() + 0.5) * zh, (rr.mean() + 0.5) * zv), rad, fill=False,
                                        ec='#F0E442', lw=1.6))
                    any_circle = True
            if i == 0:
                ax.set_title(title, fontsize=15)
                if j == 0:
                    for xt, ha, lab_ in [(0.01, 'left', 'R'), (0.99, 'right', 'L')]:
                        ax.text(xt, 0.5, lab_, transform=ax.transAxes, ha=ha, va='center', color='white',
                                fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if j == 0 and len(rows) > 1:
                ax.set_ylabel(label, fontsize=11)
    fig.text(0.01, 1 - 0.05 / H, info, va='top', ha='left', fontsize=11)

    handles = [plt.Line2D([], [], color=mf.CLASS_COLOR[c], lw=1.5, label=f'{c} renal sinus fat') for c in mf.CLASSES]
    if any_circle:
        handles.append(plt.Line2D([], [], color='#F0E442', lw=1.6, label=f'part > {thr:g} mm from other mask'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=11.5)
    fig.subplots_adjust(left=0.02, right=0.995, top=1 - top_in / H, bottom=bot_in / H, wspace=0.03, hspace=0.06)
    mf.save(fig, base, formats, dpi)


def fragment_figure(row, cls, raw, pred_dir, thr, margin, min_size, plane, anonymise, base, formats, dpi,
                    width=9.0):
    """Two rows: the slice through the main structure, and the slice with most voxels that are
    more than thr mm away from the other mask (the separate fragment). Columns: fat image,
    reference, prediction."""
    case = row['case']
    k = CLASS_ID[cls]
    img, zooms = load_xyz(os.path.join(raw, 'imagesTr', f'{case}_000{BG[0]}.nii.gz'), np.float32)
    ref = np.rint(load_xyz(os.path.join(raw, 'labelsTr', f'{case}.nii.gz'))[0]).astype(int)
    pred = np.rint(load_xyz(os.path.join(pred_dir, f'{case}.nii.gz'))[0]).astype(int)
    if not (img.shape == ref.shape == pred.shape):
        raise SystemExit(f'{case}: image {img.shape}, label {ref.shape} and prediction {pred.shape} differ in shape')
    sum_axes, aspect = plane_info(plane, zooms)

    union = (ref == k) | (pred == k)
    s_main = int(union.sum(axis=sum_axes).argmax())
    panels = [('main', s_main, plane_slice(union, plane, s_main))]

    far = np.zeros(ref.shape, bool)
    for t in (thr, thr / 2):
        far = far_voxels(ref == k, pred == k, zooms, t)
        if far.any():
            break
    if far.any():
        s_far = int(far.sum(axis=sum_axes).argmax())
        panels.append(('far', s_far, plane_slice(far, plane, s_far)))

    prepared = []
    for kind, s, m in panels:
        if not m.any():
            m = plane_slice((ref > 0) | (pred > 0), plane, s)
            if not m.any():
                continue
        r0, r1, c0, c1 = crop_box(m, margin, min_size)
        crop = plane_slice(img, plane, s)[r0:r1, c0:c1]
        prepared.append(dict(kind=kind, s=s, box=(r0, r1, c0, c1), crop=crop,
                             vmax=max(np.percentile(crop, 99.5), 1e-6),
                             h=crop.shape[0] * aspect / crop.shape[1]))

    first = [f'{cls.capitalize()}: Dice {row[f"dice_{cls}"]:.2f}']
    if np.isfinite(row.get(f'hd95_mm_{cls}', np.nan)):
        first[0] += f', HD95 {row[f"hd95_mm_{cls}"]:.0f} mm'
    first.append(f'ref {row[f"vol_ref_ml_{cls}"]:.1f} mL, pred {row[f"vol_pred_ml_{cls}"]:.1f} mL')
    if not anonymise:
        first.append(case)
    labels = {'main': first, 'far': ['Slice with the separate fragment']}

    col_w = width / 3.05
    heights = [col_w * d['h'] for d in prepared]
    fig, axes = plt.subplots(len(prepared), 3, figsize=(width, sum(heights) + 0.9 + 0.1 * len(prepared)),
                             squeeze=False, gridspec_kw=dict(height_ratios=heights, hspace=0.04, wspace=0.03))
    for i, d in enumerate(prepared):
        r0, r1, c0, c1 = d['box']
        for j, (title, mask) in enumerate([(BG[1], None), ('Reference', ref), ('Prediction', pred)]):
            ax = axes[i, j]
            ax.imshow(d['crop'], cmap='gray', vmin=0, vmax=d['vmax'], origin='lower', aspect=aspect)
            if mask is not None:
                for kk, name in enumerate(mf.CLASSES, start=1):
                    sub = (plane_slice(mask, plane, d['s'])[r0:r1, c0:c1] == kk)
                    if sub.any() and not sub.all():
                        ax.contour(sub.astype(float), levels=[0.5], colors=[mf.CLASS_COLOR[name]], linewidths=1.3)
            if j == 0:
                for xt, ha, lab in [(0.02, 'left', 'R'), (0.98, 'right', 'L')]:
                    ax.text(xt, 0.5, lab, transform=ax.transAxes, ha=ha, va='center', color='white',
                            fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=15)
        axes[i, 0].text(0.02, 0.97, '\n'.join(labels[d['kind']]), transform=axes[i, 0].transAxes, va='top',
                        ha='left', fontsize=11, color='white', bbox=dict(facecolor='black', alpha=0.6, pad=2, lw=0))
    handles = [plt.Line2D([], [], color=mf.CLASS_COLOR[c], lw=1.5, label=f'{c} renal sinus fat') for c in mf.CLASSES]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.94, bottom=0.07)
    mf.save(fig, base, formats, dpi)


def small_overview_figure(row, cls, raw, pred_dir, plane, anonymise, base, formats, dpi, panel_width=5.2,
                          marker=True):
    """Case on the whole field of view: reference and prediction side by side, in the slice where
    the structure (reference + prediction) is largest. Contours of both classes are drawn on that
    slice only. With marker=True (small volumes) a yellow circle marks the structure of interest;
    with marker=False (large volumes) there is no circle and volumes are given with 1 decimal."""
    from matplotlib.patches import Circle
    case = row['case']
    k = CLASS_ID[cls]
    img, zooms = load_xyz(os.path.join(raw, 'imagesTr', f'{case}_000{BG[0]}.nii.gz'), np.float32)
    ref = np.rint(load_xyz(os.path.join(raw, 'labelsTr', f'{case}.nii.gz'))[0]).astype(int)
    pred = np.rint(load_xyz(os.path.join(pred_dir, f'{case}.nii.gz'))[0]).astype(int)
    if not (img.shape == ref.shape == pred.shape):
        raise SystemExit(f'{case}: image {img.shape}, label {ref.shape} and prediction {pred.shape} differ in shape')
    zx, zy, zz = zooms
    sum_axes, _ = plane_info(plane, zooms)
    zh, zv = (zx, zz) if plane == 'coronal' else (zx, zy)

    target = (ref == k) | (pred == k)
    if not target.any():
        target = (ref > 0) | (pred > 0)
    s = int(target.sum(axis=sum_axes).argmax())

    bg = plane_slice(img, plane, s)
    nr, nc = bg.shape
    xs, ys = (np.arange(nc) + 0.5) * zh, (np.arange(nr) + 0.5) * zv
    vmax = max(np.percentile(bg, 99.5), 1e-6)
    t2 = plane_slice(target, plane, s)
    circle = None
    if marker and t2.any():
        rr, cc = np.where(t2)
        circle = ((cc.mean() + 0.5) * zh, (rr.mean() + 0.5) * zv,
                  max(10.0, 0.5 * max(np.ptp(cc) * zh, np.ptp(rr) * zv) + 8.0))

    dec = 2 if marker else 1
    info = f'{cls.capitalize()}: Dice {row[f"dice_{cls}"]:.2f}'
    if np.isfinite(row.get(f'hd95_mm_{cls}', np.nan)):
        info += f', HD95 {row[f"hd95_mm_{cls}"]:.1f} mm'
    info += f'\nref {row[f"vol_ref_ml_{cls}"]:.{dec}f} mL, pred {row[f"vol_pred_ml_{cls}"]:.{dec}f} mL'
    if not anonymise:
        info += f'\n{case}'
    top_in, bot_in = 0.45 + 0.2 * (info.count('\n') + 1), 0.7
    H = panel_width * (nr * zv) / (nc * zh) + top_in + bot_in

    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_width, H))
    for j, (ax, title, mask) in enumerate([(axes[0], 'Reference', ref), (axes[1], 'Prediction', pred)]):
        ax.imshow(bg, cmap='gray', vmin=0, vmax=vmax, origin='lower', extent=[0, nc * zh, 0, nr * zv],
                  aspect='equal')
        for kk, name in enumerate(mf.CLASSES, start=1):
            m2 = plane_slice(mask == kk, plane, s)
            if m2.any() and not m2.all():
                ax.contour(xs, ys, m2.astype(float), levels=[0.5], colors=[mf.CLASS_COLOR[name]], linewidths=1.4)
        if circle:
            ax.add_patch(Circle(circle[:2], circle[2], fill=False, ec='#F0E442', lw=1.6))
        if j == 0:
            for xt, ha, lab_ in [(0.01, 'left', 'R'), (0.99, 'right', 'L')]:
                ax.text(xt, 0.5, lab_, transform=ax.transAxes, ha=ha, va='center', color='white',
                        fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.01, 1 - 0.05 / H, info, va='top', ha='left', fontsize=11)
    handles = [plt.Line2D([], [], color=mf.CLASS_COLOR[c], lw=1.5, label=f'{c} renal sinus fat') for c in mf.CLASSES]
    if circle:
        handles.append(plt.Line2D([], [], color='#F0E442', lw=1.6, label=f'{cls} structure of interest'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=11.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - top_in / H, bottom=bot_in / H, wspace=0.03)
    mf.save(fig, base, formats, dpi)


def swap_figure(row, raw, pred_dir, plane, anonymise, base, formats, dpi, panel_width=5.2):
    """Class assignment error on the whole field of view: reference and prediction side by side,
    in the slice with most voxels that got the other kidney's label. Contours of both classes are
    drawn on that slice only; the yellow circle marks the wrongly assigned part. The text gives
    Dice, HD95 and reference/predicted volume for left and right."""
    from matplotlib.patches import Circle
    from scipy import ndimage
    case = row['case']
    img, zooms = load_xyz(os.path.join(raw, 'imagesTr', f'{case}_000{BG[0]}.nii.gz'), np.float32)
    ref = np.rint(load_xyz(os.path.join(raw, 'labelsTr', f'{case}.nii.gz'))[0]).astype(int)
    pred = np.rint(load_xyz(os.path.join(pred_dir, f'{case}.nii.gz'))[0]).astype(int)
    if not (img.shape == ref.shape == pred.shape):
        raise SystemExit(f'{case}: image {img.shape}, label {ref.shape} and prediction {pred.shape} differ in shape')
    zx, zy, zz = zooms
    sum_axes, _ = plane_info(plane, zooms)
    zh, zv = (zx, zz) if plane == 'coronal' else (zx, zy)

    wrong = ((pred == 1) & (ref == 2)) | ((pred == 2) & (ref == 1))     # label of the other kidney
    if wrong.any():
        s = int(wrong.sum(axis=sum_axes).argmax())
    else:
        print(f'  {case}: no voxels with the other kidney label; showing the slice with both kidneys')
        a_l = ((ref == 1) | (pred == 1)).sum(axis=sum_axes)
        a_r = ((ref == 2) | (pred == 2)).sum(axis=sum_axes)
        s = int((np.minimum(a_l, a_r) * 1000 + a_l + a_r).argmax())
    lab, n = ndimage.label(wrong)
    comps = [int(c) + 1 for c in (np.argsort(ndimage.sum(wrong, lab, range(1, n + 1)))[::-1][:2] if n else [])]

    bg = plane_slice(img, plane, s)
    nr, nc = bg.shape
    xs, ys = (np.arange(nc) + 0.5) * zh, (np.arange(nr) + 0.5) * zv
    vmax = max(np.percentile(bg, 99.5), 1e-6)

    lines = []
    for cls in mf.CLASSES:
        t = f'{cls.capitalize()}: Dice {row[f"dice_{cls}"]:.2f}'
        if np.isfinite(row.get(f'hd95_mm_{cls}', np.nan)):
            t += f', HD95 {row[f"hd95_mm_{cls}"]:.1f} mm'
        t += f', ref {row[f"vol_ref_ml_{cls}"]:.1f} mL, pred {row[f"vol_pred_ml_{cls}"]:.1f} mL'
        lines.append(t)
    if not anonymise:
        lines.append(case)
    info = '\n'.join(lines)
    top_in, bot_in = 0.45 + 0.2 * len(lines), 0.7
    H = panel_width * (nr * zv) / (nc * zh) + top_in + bot_in

    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_width, H))
    any_circle = False
    for j, (ax, title, mask) in enumerate([(axes[0], 'Reference', ref), (axes[1], 'Prediction', pred)]):
        ax.imshow(bg, cmap='gray', vmin=0, vmax=vmax, origin='lower', extent=[0, nc * zh, 0, nr * zv],
                  aspect='equal')
        for kk, name in enumerate(mf.CLASSES, start=1):
            m2 = plane_slice(mask == kk, plane, s)
            if m2.any() and not m2.all():
                ax.contour(xs, ys, m2.astype(float), levels=[0.5], colors=[mf.CLASS_COLOR[name]], linewidths=1.4)
        for comp in comps:
            c2 = plane_slice(lab == comp, plane, s)
            if c2.any():
                rr, cc = np.where(c2)
                rad = max(10.0, 0.5 * max(np.ptp(cc) * zh, np.ptp(rr) * zv) + 8.0)
                ax.add_patch(Circle(((cc.mean() + 0.5) * zh, (rr.mean() + 0.5) * zv), rad, fill=False,
                                    ec='#F0E442', lw=1.6))
                any_circle = True
        if j == 0:
            for xt, ha, lab_ in [(0.01, 'left', 'R'), (0.99, 'right', 'L')]:
                ax.text(xt, 0.5, lab_, transform=ax.transAxes, ha=ha, va='center', color='white',
                        fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.01, 1 - 0.05 / H, info, va='top', ha='left', fontsize=11)
    handles = [plt.Line2D([], [], color=mf.CLASS_COLOR[c], lw=1.5, label=f'{c} renal sinus fat') for c in mf.CLASSES]
    if any_circle:
        handles.append(plt.Line2D([], [], color='#F0E442', lw=1.6, label='part with the label of the other kidney'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=11.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - top_in / H, bottom=bot_in / H, wspace=0.03)
    mf.save(fig, base, formats, dpi)


def pick_median_case(df):
    """Examination whose left and right Dice are both closest to their own median: the smallest
    largest deviation of the two (in Dice units). Only examinations with both Dice defined."""
    d = df[df['dice_left'].notna() & df['dice_right'].notna()]
    med_l, med_r = d['dice_left'].median(), d['dice_right'].median()
    dev = np.maximum((d['dice_left'] - med_l).abs(), (d['dice_right'] - med_r).abs())
    return d.loc[dev.idxmin()], med_l, med_r


def median_figure(row, raw, pred_dir, plane, anonymise, base, formats, dpi, panel_width=5.2):
    """Example with Dice close to the median, no markers: whole field of view, reference and
    prediction side by side, in the slice where both kidneys are best visible. The text gives
    Dice, HD95 and reference/predicted volume for left and right."""
    case = row['case']
    img, zooms = load_xyz(os.path.join(raw, 'imagesTr', f'{case}_000{BG[0]}.nii.gz'), np.float32)
    ref = np.rint(load_xyz(os.path.join(raw, 'labelsTr', f'{case}.nii.gz'))[0]).astype(int)
    pred = np.rint(load_xyz(os.path.join(pred_dir, f'{case}.nii.gz'))[0]).astype(int)
    if not (img.shape == ref.shape == pred.shape):
        raise SystemExit(f'{case}: image {img.shape}, label {ref.shape} and prediction {pred.shape} differ in shape')
    zx, zy, zz = zooms
    sum_axes, _ = plane_info(plane, zooms)
    zh, zv = (zx, zz) if plane == 'coronal' else (zx, zy)

    a_l = ((ref == 1) | (pred == 1)).sum(axis=sum_axes)
    a_r = ((ref == 2) | (pred == 2)).sum(axis=sum_axes)
    s = int((np.minimum(a_l, a_r) * 1000 + a_l + a_r).argmax())

    bg = plane_slice(img, plane, s)
    nr, nc = bg.shape
    xs, ys = (np.arange(nc) + 0.5) * zh, (np.arange(nr) + 0.5) * zv
    vmax = max(np.percentile(bg, 99.5), 1e-6)

    lines = []
    for cls in mf.CLASSES:
        t = f'{cls.capitalize()}: Dice {row[f"dice_{cls}"]:.2f}'
        if np.isfinite(row.get(f'hd95_mm_{cls}', np.nan)):
            t += f', HD95 {row[f"hd95_mm_{cls}"]:.1f} mm'
        t += f', ref {row[f"vol_ref_ml_{cls}"]:.1f} mL, pred {row[f"vol_pred_ml_{cls}"]:.1f} mL'
        lines.append(t)
    if not anonymise:
        lines.append(case)
    info = '\n'.join(lines)
    top_in, bot_in = 0.45 + 0.2 * len(lines), 0.7
    H = panel_width * (nr * zv) / (nc * zh) + top_in + bot_in

    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_width, H))
    for j, (ax, title, mask) in enumerate([(axes[0], 'Reference', ref), (axes[1], 'Prediction', pred)]):
        ax.imshow(bg, cmap='gray', vmin=0, vmax=vmax, origin='lower', extent=[0, nc * zh, 0, nr * zv],
                  aspect='equal')
        for kk, name in enumerate(mf.CLASSES, start=1):
            m2 = plane_slice(mask == kk, plane, s)
            if m2.any() and not m2.all():
                ax.contour(xs, ys, m2.astype(float), levels=[0.5], colors=[mf.CLASS_COLOR[name]], linewidths=1.4)
        if j == 0:
            for xt, ha, lab_ in [(0.01, 'left', 'R'), (0.99, 'right', 'L')]:
                ax.text(xt, 0.5, lab_, transform=ax.transAxes, ha=ha, va='center', color='white',
                        fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.text(0.01, 1 - 0.05 / H, info, va='top', ha='left', fontsize=11)
    handles = [plt.Line2D([], [], color=mf.CLASS_COLOR[c], lw=1.5, label=f'{c} renal sinus fat') for c in mf.CLASSES]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=11.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - top_in / H, bottom=bot_in / H, wspace=0.03)
    mf.save(fig, base, formats, dpi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xlsx', required=True, help='results_summary.xlsx made by results_summary.py')
    p.add_argument('--out', default=None, help='Output folder (default: figures next to the xlsx)')
    p.add_argument('--formats', nargs='+', default=['png', 'pdf'])
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--max-ref-ml', type=float, default=2.0,
                   help='Make a figure per examination with a reference volume below this (mL)')
    p.add_argument('--min-ref-ml', type=float, default=None,
                   help='Also make a figure per examination with a reference volume above this (mL), '
                        'e.g. 40 (files large_<side>_<case>, sorted from large to small volume)')
    p.add_argument('--linear', action='store_true',
                   help='Also make fig_dice_vs_volume_linear (linear x axis) next to the log version')
    p.add_argument('--hd95-mm', type=float, default=20.0,
                   help='Threshold for a large HD95 (mm) in fig_hd95_vs_dice and the fragment figures')
    p.add_argument('--median', action='store_true',
                   help='With --build: figure of the examination whose left and right Dice are both closest '
                        'to their median (no markers)')
    p.add_argument('--median-case', default=None, help='Case id for --median instead of the automatic selection')
    p.add_argument('--swap', action='store_true',
                   help='With --build: figure of the examination(s) where a kidney got the label of the other '
                        'kidney (default selection: reference volume >= --swap-min-ref-ml and Dice < 0.5)')
    p.add_argument('--swap-cases', nargs='+', default=None, help='Case ids for --swap instead of the automatic selection')
    p.add_argument('--swap-min-ref-ml', type=float, default=1.0)
    p.add_argument('--fragments', action='store_true',
                   help='With --build: one figure per examination (and class) with HD95 above --hd95-mm')
    p.add_argument('--zoom', action='store_true',
                   help='Fragment figures: use the zoomed two-row version instead of the overview picture')
    p.add_argument('--fragments-min-dice', type=float, default=0.0,
                   help='Only make fragment figures for examinations with Dice >= this (e.g. 0.8)')
    p.add_argument('--background', choices=list(CHANNELS), default='fat',
                   help='Image shown behind the masks in the case figures: fat image, fat fraction map (ff) or water')
    p.add_argument('--side', choices=['left', 'right', 'both'], default='right')
    p.add_argument('--plane', choices=['coronal', 'axial'], default='coronal',
                   help='Slice orientation of the per-examination figures (default: coronal)')
    p.add_argument('--build', default=None, help='iBEAt_Build folder (needed for the per-examination figures)')
    p.add_argument('--margin', type=int, default=12, help='Margin (voxels) around the masks in the crop')
    p.add_argument('--min-size', type=int, default=48,
                   help='Minimum crop size in voxels, so that very small masks keep some context')
    p.add_argument('--anonymise', action='store_true', help='Do not print case ids in figures and file names')
    args = p.parse_args()
    BG[:] = CHANNELS[args.background]

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.xlsx)), 'figures')
    os.makedirs(out, exist_ok=True)
    df = pd.read_excel(args.xlsx, sheet_name='Per_case')

    dice_vs_volume_figure(df, os.path.join(out, 'fig_dice_vs_volume'), args.formats, args.dpi, args.max_ref_ml)
    if args.linear:
        dice_vs_volume_figure(df, os.path.join(out, 'fig_dice_vs_volume_linear'), args.formats, args.dpi,
                              args.max_ref_ml, log=False)

    if 'hd95_mm_left' in df.columns:
        hd95_vs_dice_figure(df, os.path.join(out, 'fig_hd95_vs_dice'), args.formats, args.dpi, args.hd95_mm)
    else:
        print('No HD95 columns in Per_case (run results_summary.py with --surface): skipping HD95 figures.')

    if not args.build:
        print('No --build given: skipping the per-examination figures.')
        return
    raw = os.path.join(args.build, mf.PIPELINE, 'stage_8_build_training_data', mf.DATASET)
    pred_dir = os.path.join(args.build, mf.PIPELINE, 'stage_10_train', mf.DATASET, mf.TRAINER, mf.CROSSVAL)

    if args.median:
        if args.median_case:
            hit = df[df['case'] == args.median_case]
            if hit.empty:
                raise SystemExit(f'Case {args.median_case} not found in Per_case')
            row = hit.iloc[0]
        else:
            row, med_l, med_r = pick_median_case(df)
            print(f'Median Dice: left {med_l:.3f}, right {med_r:.3f}')
        print(f'Median example: {row["case"]}  Dice L {row["dice_left"]:.3f}, R {row["dice_right"]:.3f}')
        tag = 'example' if args.anonymise else str(row['case'])
        median_figure(row, raw, pred_dir, args.plane, args.anonymise, os.path.join(out, f'median_{tag}'),
                      args.formats, args.dpi)

    if args.swap:
        if args.swap_cases:
            sel = df[df['case'].isin(args.swap_cases)]
            missing = set(args.swap_cases) - set(sel['case'])
            if missing:
                raise SystemExit(f'Cases not found in Per_case: {sorted(missing)}')
        else:
            hit = np.zeros(len(df), bool)
            for cls in mf.CLASSES:
                hit |= ((df[f'vol_ref_ml_{cls}'] >= args.swap_min_ref_ml) & (df[f'dice_{cls}'] < 0.5)).to_numpy()
            sel = df[hit]
        print(f'Class assignment figures: {len(sel)} examination(s)')
        if sel.empty:
            print('  none found; give case ids with --swap-cases')
        for i, (_, row) in enumerate(sel.iterrows(), start=1):
            tag = f'{i:02d}' if args.anonymise else str(row['case'])
            print(f'  {i:02d}  {row["case"]}  Dice L {row["dice_left"]:.2f}, R {row["dice_right"]:.2f}')
            swap_figure(row, raw, pred_dir, args.plane, args.anonymise, os.path.join(out, f'swap_{tag}'),
                        args.formats, args.dpi)

    if args.fragments and 'hd95_mm_left' in df.columns:
        for cls in mf.CLASSES:
            big = df[(df[f'hd95_mm_{cls}'] > args.hd95_mm) & (df[f'dice_{cls}'] >= args.fragments_min_dice)]
            big = big.sort_values(f'hd95_mm_{cls}', ascending=False).reset_index(drop=True)
            print(f'{cls}: {len(big)} examinations with HD95 > {args.hd95_mm:g} mm '
                  f'(Dice >= {args.fragments_min_dice:g})')
            for i, (_, row) in enumerate(big.iterrows(), start=1):
                tag = f'{i:02d}' if args.anonymise else str(row['case'])
                print(f'  {i:02d}  {row["case"]}  HD95 {row[f"hd95_mm_{cls}"]:.0f} mm  Dice {row[f"dice_{cls}"]:.2f}')
                base = os.path.join(out, f'hd95_{cls}_{tag}')
                if args.zoom:
                    fragment_figure(row, cls, raw, pred_dir, args.hd95_mm, args.margin, args.min_size,
                                    args.plane, args.anonymise, base, args.formats, args.dpi)
                else:
                    overview_figure(row, cls, raw, pred_dir, args.hd95_mm, args.plane, args.anonymise,
                                    base, args.formats, args.dpi)

    sides = mf.CLASSES if args.side == 'both' else [args.side]

    if args.min_ref_ml is not None:
        print(f'Large-volume figures: full field of view, {args.plane}, background = {BG[1]}')
        for cls in sides:
            large = df[(df[f'vol_ref_ml_{cls}'] > args.min_ref_ml) & df[f'dice_{cls}'].notna()]
            large = large.sort_values(f'vol_ref_ml_{cls}', ascending=False).reset_index(drop=True)
            print(f'{cls}: {len(large)} of {int(df[f"dice_{cls}"].notna().sum())} examinations '
                  f'with reference volume > {args.min_ref_ml:g} mL')
            for i, (_, row) in enumerate(large.iterrows(), start=1):
                tag = f'{i:02d}' if args.anonymise else str(row['case'])
                print(f'  {i:02d}  {row["case"]}  ref {row[f"vol_ref_ml_{cls}"]:.1f} mL  '
                      f'Dice {row[f"dice_{cls}"]:.2f}')
                small_overview_figure(row, cls, raw, pred_dir, args.plane, args.anonymise,
                                      os.path.join(out, f'large_{cls}_{tag}'),
                                      args.formats, args.dpi, marker=False)

    print(f'Case figures: {"zoomed crop (--zoom)" if args.zoom else "full field of view, reference and prediction side by side"}, '
          f'{args.plane}, background = {BG[1]}')
    for cls in sides:
        small = df[(df[f'vol_ref_ml_{cls}'] < args.max_ref_ml) & df[f'dice_{cls}'].notna()]
        small = small.sort_values(f'vol_ref_ml_{cls}').reset_index(drop=True)
        print(f'{cls}: {len(small)} of {int(df[f"dice_{cls}"].notna().sum())} examinations '
              f'with reference volume < {args.max_ref_ml:g} mL')
        for i, (_, row) in enumerate(small.iterrows(), start=1):
            tag = f'{i:02d}' if args.anonymise else str(row['case'])
            print(f'  {i:02d}  {row["case"]}  ref {row[f"vol_ref_ml_{cls}"]:.2f} mL  Dice {row[f"dice_{cls}"]:.2f}')
            base = os.path.join(out, f'small_{cls}_{tag}')
            if args.zoom:
                case_figure(row, cls, raw, pred_dir, args.margin, args.min_size, args.anonymise,
                            base, args.formats, args.dpi, plane=args.plane)
            else:
                small_overview_figure(row, cls, raw, pred_dir, args.plane, args.anonymise,
                                      base, args.formats, args.dpi)


if __name__ == '__main__':
    main()