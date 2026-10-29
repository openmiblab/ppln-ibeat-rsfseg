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
PREFETCH = 2  # aantal beelden dat alvast wordt ingelezen

# Pogingen en wachttijd (s) bij een tijdelijk geblokkeerde index.json
RETRIES = 6
RETRY_DELAY = 2.0

# Alle toegang tot de data-/mask-databases (index.json) loopt via dit slot.
# dbdicom schrijft index.json bij elke close(), en write_volume(ref=...) opent
# de data-database opnieuw. Zonder slot lezen reader en writer tegelijk en
# schrijven ze tegelijk dezelfde index.json -> half bestand / toegangsfout.
DB_LOCK = threading.RLock()

# Eén geopende database per pad (alleen voor lezen; wordt nooit teruggeschreven)
_READ_DBS = {}


def _retry(fn, what, attempts=RETRIES, delay=RETRY_DELAY):
    """Voer fn uit; probeer opnieuw bij een tijdelijk geblokkeerd bestand."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except (OSError, json.JSONDecodeError) as e:
            if i == attempts:
                raise
            logging.warning(
                f"{what}: poging {i}/{attempts} mislukt ({e}); "
                f"opnieuw over {delay * i:.0f}s"
            )
            time.sleep(delay * i)


def _read_db(path):
    """Geef een gecachte, alleen-lezen database terug (leest index.json 1x)."""
    with DB_LOCK:
        dbd = _READ_DBS.get(path)
        if dbd is None:
            dbd = _retry(lambda: db.open(path), f"openen {path}")
            _READ_DBS[path] = dbd
        return dbd


def _key(s):
    """Sleutel zonder volgnummers: (patient, studienaam, serienaam).

    s heeft de vorm [pad, patient, (study, n), (series, m)]. De volgnummers
    (n, m) kunnen per database verschillen, dus daar vergelijken we niet op.
    """
    return (str(s[1]), s[2][0], s[3][0])


def run(build, logfile, plane='coronal'):
    datapath = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data')
    #datapath = r"X:\abdominal_imaging\Shared\Benthe\dixon\stage_5_clean_dixon_data"
    maskpath = os.path.join(build, 'rsfseg', 'stage_1_segment')
    editpath = os.path.join(build, 'rsfseg', 'stage_4_edit')

    # Stap 1: bouw één keer de volledige takenlijst
    sites = [('Controls',)] + [('Patients', s) for s in SITES]
    jobs = []
    for parts in sites:
        for task in TASKS:
            jobs += collect_jobs(
                os.path.join(datapath, *parts),
                os.path.join(maskpath, *parts),
                os.path.join(editpath, *parts),
                task,
            )
    logging.info(f"{len(jobs)} maskers te bewerken")

    # Stap 2: bewerken, met inlezen en schrijven op de achtergrond
    process(jobs, plane)


def collect_jobs(sitedatapath, sitemaskpath, siteeditpath, task):
    siteeditpath = os.path.join(siteeditpath, task)
    os.makedirs(siteeditpath, exist_ok=True)

    record = data.dixon_record()

    # Elke database slechts één keer openen en één keer uitlezen
    all_series = _read_db(sitedatapath).series()
    all_series_keys = {repr(s) for s in all_series}
    all_out_phase = [s for s in all_series
                     if s[3][0].endswith('out_phase')]
    out_phase_series = [s for s in all_out_phase if s[3][1] == 0]
    n_dup = len(all_out_phase) - len(out_phase_series)
    if n_dup:
        logging.info(f"{sitedatapath}: {n_dup} dubbele out-phase series (index > 0) overgeslagen")

    mask_all = _read_db(sitemaskpath).series()
    masks = {(str(s[1]), s[2][0]): s
             for s in mask_all if s[3][0] == task}
    try:
        done = {repr(s) for s in _read_db(siteeditpath).series()}
    except Exception:
        done = set()  # edit-database bestaat nog niet
    names = {s[3][0] for s in all_series}
    logging.info(f"data series namen: {sorted(names)[:20]}")
    logging.info(f"mask series namen: {sorted({s[3][0] for s in mask_all})}")
    logging.info(f"voorbeeld mask-series: {mask_all[:2]}")
    logging.info(
        f"{sitedatapath}: {len(out_phase_series)} out-phase series, "
        f"{len(masks)} {task} in {sitemaskpath}"
    )

    jobs = []
    for series_op in out_phase_series:
        patient = series_op[1]
        study = series_op[2][0]
        sequence = series_op[3][0][:-len('_out_phase')]

        # Alleen de geselecteerde dixon-sequence
        if sequence != data.dixon_series_desc(record, patient, study):
            continue

        # Alleen als er een automatisch masker bestaat
        mask_series = masks.get((str(patient), study))
        if mask_series is None:
            continue

        edited_mask_series = [siteeditpath, patient, (study, 0), (f"{task}_edited", 0)]
        if repr(edited_mask_series) in done:
            continue

        # Zoals in stage_1: fat is de referentie; water is de terugval-afbeelding
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
    return jobs


def _read_volume(series):
    """Lees een volume via de gecachte database (schrijft index.json niet)."""
    with DB_LOCK:
        dbd = _read_db(series[0])
        return dbd.volume(series, None, 0)


def load_job(job):
    """Draait in de reader-thread: alleen data- en mask-database."""
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
    """Draait in de writer-thread: schrijft naar de edit-database.

    write_volume opent met ref=... ook de data-database en schrijft daar
    index.json terug. Dat gebeurt onder DB_LOCK zodat de reader-thread niet
    tegelijk dezelfde index.json leest of schrijft.
    """
    dbd = _retry(lambda: db.open(edited_series[0]), f"openen {edited_series[0]}")
    with DB_LOCK:
        dbd.write_volume(vol, edited_series, ref=ref, verbose=0)
    # index.json van de edit-database wegschrijven, met herpogingen
    # (close() is herhaalbaar zonder gevolgen)
    _retry(dbd.close, f"schrijven index.json {edited_series[0]}")

    # Melding pas nadat het masker echt is opgeslagen
    msg = (
        f"Opgeslagen: patient {edited_series[1]}, studie {edited_series[2][0]} "
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
            refill()  # volgende beelden alvast inlezen tijdens het editen

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

        # Wachten tot alles is weggeschreven en fouten zichtbaar maken
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