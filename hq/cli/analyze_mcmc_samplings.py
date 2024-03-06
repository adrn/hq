import os

import pytensor

pytensor.config.optimizer = "None"
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.reoptimize_unpickled_function = False
pytensor.config.cxx = ""
import arviz as az
import h5py
import numpy as np
import thejoker as tj
from astropy.table import QTable, Table, vstack
from thejoker.multiproc_helpers import batch_tasks
from thejoker.samples_helpers import inferencedata_to_samples
from tqdm import tqdm

from hq.config import Config
from hq.log import logger

from .analyze_joker_samplings import compute_metadata


def worker(task):
    source_ids, worker_id, conf = task

    logger.debug(f"Worker {worker_id}: {len(source_ids)} stars left to process")
    prior, _ = conf.get_prior()

    rows = []
    sub_samples = {}
    units = None
    for source_id in source_ids:
        with h5py.File(conf.tasks_file, "r") as tasks_f:
            data = tj.RVData.from_timeseries(tasks_f[source_id])

        source_path = conf.cache_path / "mcmc" / str(source_id)
        init_mcmc_file = source_path / "init-samples.nc"
        main_mcmc_file = source_path / "samples.nc"

        if not source_path.exists():
            logger.warning(f"{source_id}: MCMC path does not exist at {source_path}")
            continue
        elif not init_mcmc_file.exists():
            logger.warning(
                f"{source_id}: MCMC sample data not found at {init_mcmc_file}"
            )
            continue
        elif not main_mcmc_file.exists():
            logger.debug(
                f"{source_id}: main MCMC sample data not found at "
                f"{init_mcmc_file} -- defaulting to init sample data"
            )

        if main_mcmc_file.exists():
            trace = az.from_netcdf(main_mcmc_file)
            init_samples = False
        else:
            trace = az.from_netcdf(init_mcmc_file)
            init_samples = True

        samples = inferencedata_to_samples(prior, trace, data)

        row = compute_metadata(conf, samples, data, MAP_err=True)
        row[conf.source_id_colname] = source_id

        row["joker_completed"] = False
        row["mcmc_completed"] = True

        stat_df = az.summary(trace)
        row["gelman_rubin_max"] = stat_df["r_hat"].max()

        status_bits = []
        if init_samples:
            status_bits.append(1)
        if row["gelman_rubin_max"] > conf.mcmc_max_r_hat:
            status_bits.append(2)
        if len(samples) < conf.requested_samples_per_star:
            status_bits.append(3)
        row["mcmc_status"] = np.sum(2 ** np.array(status_bits)).astype(int)

        if units is None:
            units = dict()
            for k in row.keys():
                if hasattr(row[k], "unit"):
                    units[k] = row[k].unit

        for k in units:
            if hasattr(row[k], "unit"):
                row[k] = row[k].value

        rows.append(row)

        # Now write out the requested number of samples:
        idx = np.random.choice(
            len(samples),
            size=min(len(samples), conf.requested_samples_per_star),
            replace=False,
        )
        sub_samples[source_id] = samples[idx]

    tbl = Table(rows)
    return {"tbl": tbl, "samples": sub_samples, "units": units}


def analyze_mcmc_samplings(run_path, pool):
    conf = Config(run_path / "config.yml")

    source_ids = sorted(
        [x for x in os.listdir(conf.cache_path / "mcmc") if not x.startswith(".")]
    )

    tasks = batch_tasks(len(source_ids), pool.size, arr=source_ids, args=(conf,))
    logger.info(f"Done preparing tasks: {len(tasks)} stars in process queue")

    sub_tbls = []
    all_samples = {}
    for result in tqdm(pool.map(worker, tasks), total=len(tasks)):
        if result is not None:
            sub_tbls.append(result["tbl"])
            all_samples.update(result["samples"])

    # Write the MCMC metadata table
    tbl = vstack(sub_tbls)
    for k in result["units"]:
        tbl[k].unit = result["units"][k]
    tbl = QTable(tbl)
    tbl.write(conf.metadata_mcmc_file, overwrite=True)

    # Now write out all of the individual samplings:
    with h5py.File(conf.mcmc_results_file, "a") as results_f:
        for source_id, samples in all_samples.items():
            if source_id in results_f:
                del results_f[source_id]
            g = results_f.create_group(source_id)
            samples.write(g)
