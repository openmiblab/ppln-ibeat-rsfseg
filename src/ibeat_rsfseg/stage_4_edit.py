"""Editing of masks, or replacement of automasks by edited masks"""

import os
import json
import time
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import numpy as np
import dbdicom as db
from miblab import pipe

from utils import data, edit


PIPELINE = 'rsfseg'

RSF_CLASS_MAP = {
    1: 'left_kidney_sinus_fat',
    2: 'right_kidney_sinus_fat',
}

TASKS = ['rsf_masks']

SITES = ['Exeter', 'Bari', 'Leeds', 'Bordeaux', 'Turku', 'Sheffield']
PREFETCH = 2  # Number of images that are read at the same time to make it faster

RETRIES = 6
RETRY_DELAY = 2.0
DB_LOCK = threading.RLock()
_READ_DBS = {}
_WRITE_DBS = {}


class _CachedRef:
    """Stand-in for the DataBaseDicom that write_volume(ref=...) creates itself.

    On every call, dbdicom.dbd.write_volume does DataBaseDicom(ref[0]) and
    then ref_mgr.close(). That means: re-reading the entire index.json of the
    data database and writing it back again, for every single mask. If that
    index.json is then unreadable (just written, locked), dbdicom deletes it
    and re-reads the ENTIRE dicom folder.
    Here we return the already loaded register instead, and close() is a no-op.
    """
    def __init__(self, dbd):
        self.register = dbd.register

    def close(self):
        return self


def _patch_dbdicom_ref():
    import dbdicom.dbd as _dbd
    if getattr(_dbd, '_ref_patched', False):
        return
    _orig = _dbd.DataBaseDicom

    def _factory(path, *a, **k):
        with DB_LOCK:
            cached = _READ_DBS.get(path)
        if cached is not None:
            return _CachedRef(cached)
        return _orig(path, *a, **k)

    _dbd.DataBaseDicom = _factory
    _dbd._ref_patched = True


_patch_dbdicom_ref()


def _retry(fn, what, attempts=RETRIES, delay=RETRY_DELAY):
    """Run fn; retry if a file is temporarily locked."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except (OSError, json.JSONDecodeError) as e:
            if i == attempts:
                raise
            logging.warning(
                f"{what}: attempt {i}/{attempts} failed ({e}); "
                f"retrying in {delay * i:.0f}s"
            )
            time.sleep(delay * i)



def _key(s):
    """Key without sequence numbers: (patient, study name, series name).

    s has the form [path, patient, (study, n), (series, m)]. The sequence
    numbers (n, m) can differ per database, so we do not compare on those.
    """
    return (str(s[1]), s[2][0], s[3][0])


def run(build, logfile, plane='coronal'):
    datapath = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data')
    maskpath = os.path.join(build, 'rsfseg', 'stage_1_segment')
    editpath = os.path.join(build, 'rsfseg', 'stage_4_edit')

    # Step 1: build the complete task list once
   # sites = [('Controls',)] + [('Patients', s) for s in SITES]
    sites = [('Patients', s) for s in SITES]
    jobs = []
    for parts in sites:
        for task in TASKS:
            jobs += collect_jobs(
                build,
                os.path.join(datapath, *parts),
                os.path.join(maskpath, *parts),
                os.path.join(editpath, *parts),
                task,
            )
    logging.info(f"{len(jobs)} masks to edit")

    # Step 2: edit, with reading and writing in the background
    process(jobs, plane)

def _read_db(path):
    """Return a cached, read-only database (reads index.json once).

    db.open happens outside DB_LOCK, so that multiple databases can be
    loaded at the same time. The lock only protects the cache.
    """
    with DB_LOCK:
        dbd = _READ_DBS.get(path)
    if dbd is None:
        dbd = _retry(lambda: db.open(path), f"opening {path}")
        with DB_LOCK:
            dbd = _READ_DBS.setdefault(path, dbd)
    return dbd


def collect_jobs(build, sitedatapath, sitemaskpath, siteeditpath, task):
    siteeditpath = os.path.join(siteeditpath, task)
    os.makedirs(siteeditpath, exist_ok=True)

    record = data.dixon_record(build)

    # Read data, mask and edit indices in parallel (network I/O)
    def timed(label, fn):
        t0 = time.perf_counter()
        result = fn()
        logging.info(f"{label}: {time.perf_counter() - t0:.1f}s")
        return result

    def read_done():
        try:
            return {repr(s) for s in _read_db(siteeditpath).series()}
        except Exception:
            return set()  # edit database does not exist yet

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_data = ex.submit(timed, "data index",
                           lambda: _read_db(sitedatapath).series())
        f_mask = ex.submit(timed, "mask index",
                           lambda: _read_db(sitemaskpath).series())
        f_done = ex.submit(timed, "edit index", read_done)
        all_series = f_data.result()
        mask_all = f_mask.result()
        done = f_done.result()
    logging.info(f"all indices read: {time.perf_counter() - t_start:.1f}s")

    all_series_keys = {repr(s) for s in all_series}
    all_out_phase = [s for s in all_series
                     if s[3][0].endswith('out_phase')]
    out_phase_series = [s for s in all_out_phase if s[3][1] == 0]
    n_dup = len(all_out_phase) - len(out_phase_series)
    if n_dup:
        logging.info(f"{sitedatapath}: skipped {n_dup} duplicate out-phase series (index > 0)")

    masks = {(str(s[1]), s[2][0]): s
             for s in mask_all if s[3][0] == task}
    logging.info(
        f"{sitedatapath}: {len(out_phase_series)} out-phase series, "
        f"{len(masks)} {task} in {sitemaskpath}"
    )

    t_loop = time.perf_counter()
    jobs = []
    for series_op in out_phase_series:
        patient = series_op[1]
        study = series_op[2][0]
        sequence = series_op[3][0][:-len('_out_phase')]

        # Cheapest checks first: does a mask exist, and has it not been edited yet?
        mask_series = masks.get((str(patient), study))
        if mask_series is None:
            continue

        edited_mask_series = [siteeditpath, patient, (study, 0), (f"{task}_edited", 0)]
        if repr(edited_mask_series) in done:
            continue

        # Only the selected dixon sequence
        if sequence != data.dixon_series_desc(record, patient, study):
            continue

        # As in stage_1: fat is the reference; water is the fallback image
        series_fat = series_op[:3] + [(sequence + '_fat', 0)]
        if repr(series_fat) not in all_series_keys:
            logging.info(f"{patient} {study}: fat series not found, skipping.")
            continue
        series_water = series_op[:3] + [(sequence + '_water', 0)]
        if repr(series_water) not in all_series_keys:
            series_water = None

        jobs.append({
            'patient': patient,
            'study': study,
            'series_fat': series_fat,
            'series_water': series_water,
            'series_op': series_op,
            'label_series': mask_series,
            'edited_series': edited_mask_series,
        })
    logging.info(f"job loop: {time.perf_counter() - t_loop:.1f}s, {len(jobs)} jobs")
    return jobs


def _read_volume(series):
    """Read a volume via the cached database (does not write index.json)."""
    with DB_LOCK:
        dbd = _read_db(series[0])
        return dbd.volume(series, None, 0)


def load_job(job):
    """Runs in the reader thread: only the data and mask databases."""
    try:
        op = _read_volume(job['series_op'])
    except Exception as e:
        logging.warning(
            f"{job['patient']} {job['study']} error reading out-phase "
            f"{job['series_op']}: {e}; falling back to water"
        )
        if job['series_water'] is None:
            return None
        try:
            op = _read_volume(job['series_water'])
        except Exception as e2:
            logging.error(
                f"{job['patient']} {job['study']} error reading water "
                f"{job['series_water']}: {e2}"
            )
            return None
    try:
        auto_label = _read_volume(job['label_series'])
        arr = np.round(auto_label.values).astype(np.int16)
    except Exception as e:
        logging.error(
            f"{job['patient']} {job['study']} error reading mask "
            f"{job['label_series']}: {e}"
        )
        return None
    return op, arr


def write_job(vol, edited_series, ref):
    """Runs in the writer thread: writes to the edit database.

    With ref=..., write_volume also opens the data database and writes
    index.json back there. This happens under DB_LOCK so that the reader
    thread does not read or write the same index.json at the same time.
    """
    path = edited_series[0]
    dbd = _WRITE_DBS.get(path)
    if dbd is None:
        dbd = _retry(lambda: db.open(path), f"opening {path}")
        _WRITE_DBS[path] = dbd
    with DB_LOCK:
        dbd.write_volume(vol, edited_series, ref=ref, verbose=0)
    # Write the index.json of the edit database, with retries
    # (close() can safely be called repeatedly)
    _retry(dbd.close, f"writing index.json {edited_series[0]}")

    # Report only after the mask has actually been saved
    msg = (
        f"Saved: patient {edited_series[1]}, study {edited_series[2][0]} "
        f"-> {edited_series[0]}"
    )
    logging.info(msg)
    print(msg, flush=True)


def process(jobs, plane):
    transp = {
        'coronal': (0, 1, 2),
        'sagittal': (2, 1, 0),
        'axial': (0, 2, 1),
    }[plane]

    write_futures = []
    with ThreadPoolExecutor(max_workers=1) as reader, \
         ThreadPoolExecutor(max_workers=1) as writer:

        queue = deque()
        it = iter(jobs)

        def refill():
            while len(queue) < PREFETCH:
                job = next(it, None)
                if job is None:
                    return
                queue.append((job, reader.submit(load_job, job)))

        refill()
        for _ in tqdm(range(len(jobs))):
            job, fut = queue.popleft()
            refill()  # start reading the next images while editing

            loaded = fut.result()
            if loaded is None:
                continue
            op, auto_label_arr = loaded

            try:
                oparr = op.values.transpose(transp)
                lblarr = auto_label_arr.transpose(transp)
                edited_label = edit.edit_mask_with_napari(oparr, lblarr, default_label=1)
                edited_label = edited_label.transpose(transp)
            except Exception as e:
                logging.error(f"{job['patient']} {job['study']} error editing mask: {e}")
            else:
                vol = (edited_label.astype(np.int16), op.affine)
                write_futures.append(
                    writer.submit(write_job, vol, job['edited_series'], job['series_fat'])
                )

        # Wait until everything has been written and surface any errors
        for wf in write_futures:
            try:
                wf.result()
            except Exception as e:
                logging.error(f"error writing mask: {e}")


if __name__ == '__main__':

    BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"
    kwargs = {
        "plane": {
            'type': str,
            'default': 'coronal',
            'help': 'Plane',
        }
    }
    pipe.run_stage(run, BUILD, PIPELINE, __file__, **kwargs)