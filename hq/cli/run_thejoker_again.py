"""
This script is used to re-run The Joker on samplings that were incomplete, but
with restricted period ranges to make the sampling more efficient. This should
be run after run_thejoker and analyzer_joker_samplings, but before run_mcmc.
"""

# Third-party
from astropy.utils import iers
iers.conf.auto_download = False
import astropy.table as at
import astropy.units as u

import theano
theano.config.optimizer = "None"
theano.config.mode = "FAST_COMPILE"
theano.config.reoptimize_unpickled_function = False
theano.config.cxx = ""
import h5py
import numpy as np
import thejoker as tj

# Project
from hq.log import logger
from hq.config import Config


def worker(task):
    conf = task['conf']
    source_id = task[conf.source_id_colname]
    prefix = f"Worker {task['idx']} ({source_id}): "

    # Load the prior:
    logger.debug(prefix + "Creating JokerPrior instance...")
    prior, model = conf.get_prior(P_min=task['P_range'][0],
                                  P_max=task['P_range'][1])

    rnd = np.random.default_rng(task['seed'])
    logger.log(1, prefix + f"Creating TheJoker instance with {rnd}")
    joker = tj.TheJoker(prior, random_state=rnd)

    rerun_samples = joker.rejection_sample(
        task['data'],
        prior_samples=conf.rerun_n_prior_samples,
        return_logprobs=True,
        in_memory=True)

    # Ensure only positive K values
    rerun_samples.wrap_K()

    result = {
        "conf": conf,
        "source_id": source_id,
        "log_prefix": prefix,
        "samples": rerun_samples
    }

    if len(rerun_samples) > task['n_joker_samples']:
        return result

    else:
        logger.warn(f"{prefix} Re-running TheJoker produced fewer samples than "
                    "in the initial batch run!")
        return None


def callback(result):
    if result is None:
        return

    conf = result["conf"]

    logger.debug(
        result["log_prefix"] + "Finished and writing results to results file"
    )

    with h5py.File(conf.joker_results_file, "a") as results_f:
        if result["source_id"] in results_f:
            del results_f[result["source_id"]]
        else:
            logger.warn("TODO")

        g = results_f.create_group(result["source_id"])
        result['samples'].write(g)


def rerun_thejoker(run_path, pool, seed=None):
    logger.debug(f"Processing pool has size = {pool.size}")

    conf = Config(run_path / "config.yml")

    if not conf.joker_results_file.exists():
        raise IOError(
            f"Joker results file '{conf.joker_results_file!s}' does not exist! "
            "Did you run hq run_thejoker?"
        )

    if not conf.metadata_file.exists():
        raise IOError(
            f"Joker metadata file '{conf.joker_results_file!s}' does not "
            "exist! Did you run hq run_thejoker?"
        )

    # Get list of incomplete samplings from the metadata file:
    metadata = at.QTable.read(conf.metadata_file)
    incomplete = metadata[~metadata['joker_completed']]

    tasks = []
    n = 0
    for row in incomplete:
        # Load the joker samples for this object
        data, samples, _ = conf.get_data_samples(row['APOGEE_ID'])
        Ps = samples['P'].to_value(u.day)
        logP_ptp = np.log10(Ps.max()) - np.log10(Ps.min())

        if logP_ptp < conf.rerun_logP_ptp_threshold and not row['unimodal']:
            # If the spread in past samples is smaller than a configurable
            # threshold, and not unimodal, make a new (smaller) period range to
            # re-run with:
            logPs = np.array([
                np.log10(Ps.min()) - 0.2 * logP_ptp,  # HACK: 0.2 hard-coded
                np.log10(Ps.max()) + 0.2 * logP_ptp
            ])
            rerun_P_range = 10 ** logPs * u.day

        elif row['unimodal']:
            # If unimodal, re-run with a period range set by a configurable
            # factor around the mean period:
            rerun_P_range = (
                samples['P'].mean() / conf.rerun_P_factor,
                samples['P'].mean() * conf.rerun_P_factor
            )

        else:
            # It not unimodal, or the samples have a large range of periods, too
            # bad - this sampling will just be incomplete!
            continue

        tasks.append({
            'idx': n,
            'conf': conf,
            conf.source_id_colname: row[conf.source_id_colname],
            'data': data,
            'n_joker_samples': len(samples),
            'P_range': rerun_P_range
        })
        n += 1

    # Deterministic random number seeds:
    seedseq = np.random.SeedSequence(seed)
    seeds = seedseq.spawn(len(tasks))
    for t, s in zip(tasks, seeds):
        t['seed'] = s

    logger.info(f"Done preparing tasks: {len(tasks)} tasks")
    for r in pool.map(worker, tasks, callback=callback):
        pass
