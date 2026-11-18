"""Scanner and acquisition parameters from DICOM headers, grouped per scanner, for the Methods
section / Table 1 of the RSF paper.

Walks a folder of DICOM files, reads the headers only (no pixel data), keeps the Dixon series and
writes an Excel file with
    Summary     one row per scanner (manufacturer, model, software version, field strength, site):
                number of series / examinations and, per parameter, the value or the min-max range.
    Per_series  one row per series, to check the grouping and trace odd values.

Series are selected on their SeriesDescription / ProtocolName (--series-filter, regular
expression, case-insensitive). Check the printed list of descriptions and adjust it if needed.

Speed: for every folder only one file is read to see which series it holds (assumption: one series
per folder) and to which examination it belongs. Per examination only the first matching series is
kept and read completely (--all-series keeps every matching series). To also get the echo times of
the other contrasts, run once more with a different filter, e.g. --series-filter "opp|out" --out
dicom_params_opp.xlsx.

Reading problems (unreadable files, folders that cannot be opened, e.g. when the network drive drops
out) are counted and reported at the end instead of being skipped silently.

Voxel size is the RECONSTRUCTED size from the header (PixelSpacing, and the slice spacing computed
from the slice positions). The acquired resolution can be coarser when the scanner interpolates
(zero filling); acquisition_matrix and fov_mm in the output help to check that.

Usage:
    python dicom_params.py --dicom-root <folder with DICOM files>
    python dicom_params.py --dicom-root <folder> --series-filter "dixon|water|fat" --out dicom_params.xlsx
    python dicom_params.py --dicom-root <folder> --first-only      (fastest: first file per series only)

Needs: pip install pydicom
The Per_series sheet contains patient IDs and study dates unless you use --drop-ids.
"""

import os
import re
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import pydicom


def val(ds, name):
    """Header value as plain python (None if absent, list -> list of numbers/strings)."""
    v = getattr(ds, name, None)
    if v is None or v == '':
        return None
    if isinstance(v, (pydicom.multival.MultiValue, list)):
        return [float(x) if isinstance(x, (int, float, pydicom.valuerep.DSfloat, pydicom.valuerep.IS)) else str(x)
                for x in v]
    if isinstance(v, (pydicom.valuerep.DSfloat, pydicom.valuerep.IS, int, float)):
        return float(v)
    return str(v).strip()


def first(x):
    return x[0] if isinstance(x, list) and x else x


def plane_and_spacing(headers):
    """Plane from the image orientation and slice spacing from the slice positions."""
    ds = headers[0]
    iop = getattr(ds, 'ImageOrientationPatient', None)
    plane, spacing = None, None
    if iop is not None and len(iop) == 6:
        r, c = np.array(iop[:3], float), np.array(iop[3:], float)
        n = np.cross(r, c)
        plane = ['sagittal', 'coronal', 'axial'][int(np.abs(n).argmax())]
        pos = [np.array(h.ImagePositionPatient, float) @ n for h in headers
               if getattr(h, 'ImagePositionPatient', None) is not None]
        if len(pos) > 1:
            d = np.diff(np.sort(pos))
            d = d[d > 1e-3]
            if len(d):
                spacing = float(np.median(d))
    return plane, spacing


def describe_series(headers, path):
    ds = headers[0]
    ps = val(ds, 'PixelSpacing')
    plane, slice_spacing = plane_and_spacing(headers)
    rows, cols = val(ds, 'Rows'), val(ds, 'Columns')
    fov = None
    if ps and rows and cols:
        fov = f'{rows * ps[0]:.0f} x {cols * ps[1]:.0f}'
    am = val(ds, 'AcquisitionMatrix')
    if isinstance(am, list):
        am = ' x '.join(str(int(a)) for a in am if a)
    return {
        'manufacturer': val(ds, 'Manufacturer'),
        'model': val(ds, 'ManufacturerModelName'),
        'software': val(ds, 'SoftwareVersions') if not isinstance(val(ds, 'SoftwareVersions'), list)
        else ' / '.join(map(str, val(ds, 'SoftwareVersions'))),
        'field_strength_T': val(ds, 'MagneticFieldStrength'),
        'institution': val(ds, 'InstitutionName'),
        'station': val(ds, 'StationName'),
        'series_description': val(ds, 'SeriesDescription'),
        'protocol': val(ds, 'ProtocolName'),
        'scanning_sequence': val(ds, 'ScanningSequence'),
        'sequence_variant': val(ds, 'SequenceVariant'),
        'sequence_name': val(ds, 'SequenceName'),
        'acquisition_type': val(ds, 'MRAcquisitionType'),
        'plane': plane,
        'TR_ms': val(ds, 'RepetitionTime'),
        'TE_ms': val(ds, 'EchoTime'),
        'flip_angle_deg': val(ds, 'FlipAngle'),
        'bandwidth_Hz_per_px': val(ds, 'PixelBandwidth'),
        'averages': val(ds, 'NumberOfAverages'),
        'pixel_spacing_mm': f'{ps[0]:.2f} x {ps[1]:.2f}' if ps else None,
        'pixel_x_mm': ps[0] if ps else None,
        'pixel_y_mm': ps[1] if ps else None,
        'slice_thickness_mm': val(ds, 'SliceThickness'),
        'slice_spacing_mm': slice_spacing if slice_spacing is not None else val(ds, 'SpacingBetweenSlices'),
        'rows': rows,
        'columns': cols,
        'n_slices': len(headers),
        'acquisition_matrix': am,
        'fov_mm': fov,
        'phase_encoding_direction': val(ds, 'InPlanePhaseEncodingDirection'),
        'percent_sampling': val(ds, 'PercentSampling'),
        'percent_phase_fov': val(ds, 'PercentPhaseFieldOfView'),
        'parallel_imaging': val(ds, 'ParallelAcquisitionTechnique'),
        'parallel_factor': val(ds, 'ParallelReductionFactorInPlane'),
        'receive_coil': val(ds, 'ReceiveCoilName'),
        'patient_id': val(ds, 'PatientID'),
        'study_date': val(ds, 'StudyDate'),
        'study_uid': val(ds, 'StudyInstanceUID'),
        'series_uid': val(ds, 'SeriesInstanceUID'),
        'folder': path,
    }


ERRORS = Counter()     # files that could not be read, folders that could not be opened


def read_header(path):
    """MR header without pixel data, or None (not DICOM / not MR / unreadable)."""
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        ERRORS['unreadable files'] += 1
        return None
    if getattr(ds, 'Modality', None) != 'MR' or not hasattr(ds, 'SeriesInstanceUID'):
        return None
    return ds


def probe_folder(folder, files, tries=5):
    """First readable MR header in a folder (looks at a few files at most)."""
    for f in files[:tries]:
        ds = read_header(os.path.join(folder, f))
        if ds is not None:
            return ds
    return None


def read_folder(folder, files):
    """Group the readable MR headers of one folder by series."""
    by_series = {}
    for f in files:
        ds = read_header(os.path.join(folder, f))
        if ds is not None:
            by_series.setdefault(ds.SeriesInstanceUID, []).append(ds)
    return by_series


def study_key(ds):
    return str(getattr(ds, 'StudyInstanceUID', None) or f'{getattr(ds, "PatientID", "")}|{getattr(ds, "StudyDate", "")}')


def summarise(values, nd=2):
    """Single value, or min-max for numbers, or the distinct values for text."""
    v = [x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return ''
    if all(isinstance(x, (int, float)) for x in v):
        lo, hi = min(v), max(v)
        return f'{lo:.{nd}f}'.rstrip('0').rstrip('.') if lo == hi else f'{lo:.{nd}f} - {hi:.{nd}f}'.replace('.00', '')
    c = Counter(str(x) for x in v)
    items = [k for k, _ in c.most_common(4)]
    return ' | '.join(items) + (' | ...' if len(c) > 4 else '')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dicom-root', required=True, help='Folder with the DICOM files (searched recursively)')
    p.add_argument('--out', default='dicom_params.xlsx')
    p.add_argument('--series-filter', default=r'dixon|water|fat|in.?phase|out.?phase|opp',
                   help='Regex on SeriesDescription/ProtocolName of the series to keep')
    p.add_argument('--all-series', action='store_true',
                   help='Keep every matching series (default: only the first matching series of each '
                        'examination, which is enough for the scanner parameters and much faster)')
    p.add_argument('--first-only', action='store_true',
                   help='Do not read the whole series, only its first file (fastest; slice spacing then '
                        'comes from the header, number of slices = number of files in the folder)')
    p.add_argument('--drop-ids', action='store_true', help='Leave patient IDs and study dates out of the output')
    args = p.parse_args()

    def on_walk_error(e):
        ERRORS['folders that could not be opened'] += 1

    pat = re.compile(args.series_filter, re.I)
    rows, descriptions, n_dirs, seen = [], Counter(), 0, set()
    for folder, _, files in os.walk(args.dicom_root, onerror=on_walk_error):
        if not files:
            continue
        n_dirs += 1
        if n_dirs % 200 == 0:
            print(f'  {n_dirs} folders looked at, {len(rows)} series kept from {len(seen)} examinations')
        files = sorted(files)
        probe = probe_folder(folder, files)       # one file decides whether the folder is worth reading
        if probe is None:
            continue
        text = f'{getattr(probe, "SeriesDescription", "")} {getattr(probe, "ProtocolName", "")}'
        descriptions[str(getattr(probe, 'SeriesDescription', ''))] += 1
        if not pat.search(text):
            continue
        key = study_key(probe)
        if not args.all_series and key in seen:
            continue                               # this examination already has a series in the output
        seen.add(key)
        if args.first_only:
            row = describe_series([probe], folder)
            row['n_slices'] = len(files)
            rows.append(row)
        else:                                      # whole folder: slice spacing and number of slices
            for headers in read_folder(folder, files).values():
                rows.append(describe_series(headers, folder))

    print(f'{n_dirs} folders, {len(rows)} series from {len(seen)} examinations kept by the filter '
          f'{args.series_filter!r}' + ('' if args.all_series else ' (one series per examination)'))
    for what, n in ERRORS.items():
        print(f'WARNING: {n} {what}. The result may be incomplete (network connection?). Run again to check.')
    if not rows:
        print('Series descriptions found (adjust --series-filter):')
        for d, n in descriptions.most_common(40):
            print(f'  {n:5d}  {d}')
        raise SystemExit('No series kept.')

    df = pd.DataFrame(rows)
    # echo times / flip angle etc. may be lists in some headers: take the first value
    for col in ['TE_ms', 'TR_ms', 'flip_angle_deg', 'bandwidth_Hz_per_px', 'averages', 'slice_thickness_mm',
                'slice_spacing_mm', 'percent_sampling', 'percent_phase_fov', 'parallel_factor']:
        df[col] = df[col].map(first)

    keys = ['manufacturer', 'model', 'software', 'field_strength_T', 'institution']
    for k in keys:
        df[k] = df[k].fillna('unknown')
    params = ['plane', 'acquisition_type', 'sequence_name', 'TR_ms', 'TE_ms', 'flip_angle_deg',
              'bandwidth_Hz_per_px', 'averages', 'pixel_spacing_mm', 'slice_thickness_mm', 'slice_spacing_mm',
              'rows', 'columns', 'n_slices', 'acquisition_matrix', 'fov_mm', 'parallel_imaging',
              'parallel_factor', 'receive_coil', 'series_description']
    summary = []
    for key, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key))
        row['n_series'] = len(g)
        row['n_examinations'] = g['study_uid'].nunique()
        for c in params:
            row[c] = summarise(g[c].tolist(), nd=3 if c.endswith('_mm') else 2)
        summary.append(row)
    summary = pd.DataFrame(summary).sort_values(['manufacturer', 'institution', 'model'])

    if args.drop_ids:
        df = df.drop(columns=['patient_id', 'study_date', 'study_uid', 'series_uid', 'folder'])
    with pd.ExcelWriter(args.out) as xw:
        summary.to_excel(xw, sheet_name='Summary', index=False)
        df.to_excel(xw, sheet_name='Per_series', index=False)

    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_colwidth', 40)
    print('\nScanner groups (full table in the Summary sheet):')
    print(summary[keys + ['n_series', 'n_examinations']].to_string(index=False))
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()