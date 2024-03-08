import os

import astropy.table as at
import astropy.units as u
import h5py
import numpy as np
from astropy.stats import median_absolute_deviation as MAD
from tqdm import tqdm

from hq.config import Config
from hq.log import logger


def compute_metadata(c, samples, data, MAP_err=False):
    from thejoker.samples_analysis import (
        is_P_unimodal,
        max_phase_gap,
        periods_spanned,
        phase_coverage,
        phase_coverage_per_period,
    )

    row = {}
    row["n_visits"] = len(data)
    row["baseline"] = (data.t.mjd.max() - data.t.mjd.min()) * u.day

    MAP_idx = (samples["ln_prior"] + samples["ln_likelihood"]).argmax()
    MAP_sample = samples[MAP_idx]
    for k in MAP_sample.par_names:
        row["MAP_" + k] = MAP_sample[k]

    if MAP_err:
        err = dict()
        for k in samples.par_names:
            err[k] = 1.5 * MAD(samples[k])
            row[f"MAP_{k}_err"] = err[k]

    row["t_ref_bmjd"] = MAP_sample.t_ref.tcb.mjd
    row["MAP_t0_bmjd"] = MAP_sample.get_t0().tcb.mjd

    row["MAP_ln_likelihood"] = samples["ln_likelihood"][MAP_idx]
    row["MAP_ln_prior"] = samples["ln_prior"][MAP_idx]

    if len(samples) == c.requested_samples_per_star:
        row["joker_completed"] = True
    else:
        row["joker_completed"] = False

    if is_P_unimodal(samples, data):
        row["unimodal"] = True
    else:
        row["unimodal"] = False

    row["baseline"] = (data.t.mjd.max() - data.t.mjd.min()) * u.day
    try:
        row["max_phase_gap"] = max_phase_gap(MAP_sample, data)
        row["phase_coverage"] = phase_coverage(MAP_sample, data)
        row["periods_spanned"] = periods_spanned(MAP_sample, data)
        row["phase_coverage_per_period"] = phase_coverage_per_period(MAP_sample, data)
    except Exception:
        row["max_phase_gap"] = np.nan
        row["phase_coverage"] = np.nan
        row["periods_spanned"] = np.nan
        row["phase_coverage_per_period"] = np.nan

    # Use the max marginal likelihood sample
    lls = samples.ln_unmarginalized_likelihood(data)
    row["max_unmarginalized_ln_likelihood"] = lls.max()

    return row


def worker(task):
    import thejoker as tj

    source_ids, worker_id, c = task

    logger.debug(f"Worker {worker_id}: {len(source_ids)} stars left to process")

    all_data = {}
    with h5py.File(c.tasks_file, "r") as tasks_f:
        for source_id in source_ids:
            data = tj.RVData.from_timeseries(tasks_f[source_id])
            all_data[source_id] = data

    all_samples = {}
    with h5py.File(c.joker_results_file, "r") as results_f:
        for source_id in source_ids:
            if source_id not in results_f:
                logger.warning(f"No samples for: {source_id}")
                continue

            # Load samples from The Joker and probabilities
            samples = tj.JokerSamples.read(results_f[source_id])

            if len(samples) < 1:
                logger.warning(f"No samples for: {source_id}")
                continue

            all_samples[source_id] = samples

    rows = []
    units = None
    for source_id in all_samples:
        data = all_data[source_id]
        samples = all_samples[source_id]

        row = compute_metadata(c, samples, data)
        row[c.source_id_colname] = source_id

        if units is None:
            units = dict()
            for k in row.keys():
                if hasattr(row[k], "unit"):
                    units[k] = row[k].unit

        for k in units:
            row[k] = row[k].value

        rows.append(row)

    tbl = at.QTable(rows)

    return tbl, units


def analyze_joker_samplings(run_path, pool):
    from thejoker.multiproc_helpers import batch_tasks

    c = Config(run_path / "config.yml")

    # numbers we need to validate
    for path in [c.joker_results_file, c.tasks_file]:
        if not os.path.exists(path):
            raise OSError(
                f"File {path} does not exist! Did you run the "
                "preceding pipeline steps?"
            )

    # Get data files out of config file:
    logger.debug("Loading data...")
    source_ids = np.unique(c.data[c.source_id_colname])
    tasks = batch_tasks(len(source_ids), pool.size, arr=source_ids, args=(c,))

    logger.info(f"Done preparing tasks: {len(tasks)} batches in process queue")

    sub_tbls = []
    for tbl, units in tqdm(pool.map(worker, tasks), total=len(tasks)):
        if tbl is not None:
            sub_tbls.append(tbl)

    tbl = at.vstack(sub_tbls)
    for k in units:
        tbl[k] = tbl[k] * units[k]

    logger.log(100, f"Unimodal sources: {tbl['unimodal'].sum()}")

    tbl.write(c.metadata_joker_file, overwrite=True)
