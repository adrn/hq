import astropy.table as at
import h5py
import numpy as np
import thejoker as tj
from tqdm import tqdm

from hq.config import Config
from hq.log import logger


def worker(task):
    conf = task["conf"]

    task["output_path"].mkdir(exist_ok=True)

    with h5py.File(conf.joker_results_file, "r") as joker_f:
        with h5py.File(conf.mcmc_results_file, "r") as mcmc_f:
            for row in task["metadata"]:
                source_id = str(row[conf.source_id_colname]).strip()
                filename = task["output_path"] / f"{source_id}.fits"
                if filename.exists() and not task["overwrite"]:
                    continue

                # TODO: should this be configurable?
                if 0 < row["mcmc_status"] <= 2:
                    results_f = mcmc_f
                else:
                    results_f = joker_f

                samples = tj.JokerSamples.read(
                    results_f[f"{source_id}"], path="samples"
                )

                samples.write(str(filename), overwrite=True)


def expand_samples(run_path, pool, overwrite=False):
    conf = Config(run_path / "config.yml")
    logger.debug(f"Loaded config for run {run_path}")

    meta = at.Table.read(conf.metadata_file)
    src = at.Table.read(conf.input_data_file)
    meta = at.join(meta, src, keys=conf.source_id_colname, join_type="inner")
    meta = at.unique(meta, conf.source_id_colname)

    root_output_path = conf.cache_path / "samples"
    root_output_path.mkdir(exist_ok=True)
    logger.debug(f"Writing samples to {root_output_path!s}")

    if conf.expand_subdir_column is None:
        raise ValueError("Config item expand_subdir_column must be specified!")

    unq_subdirs = np.unique(meta[conf.expand_subdir_column])
    tasks = []
    for subdir in unq_subdirs:
        sub_meta = meta[meta[conf.expand_subdir_column] == subdir]
        output_path = root_output_path / subdir
        tasks.append(
            {
                "conf": conf,
                "output_path": output_path,
                "metadata": sub_meta,
                "overwrite": overwrite,
            }
        )

    for r in tqdm(pool.map(worker, tasks), total=len(tasks)):
        pass
