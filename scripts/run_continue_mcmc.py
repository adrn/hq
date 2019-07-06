# Standard library
import os
from os.path import join, exists
import sys
import time
import pickle

# Third-party
from astropy.table import QTable
import matplotlib.pyplot as plt
import numpy as np
from thejoker.log import log as joker_logger
from thejoker.sampler import TheJoker, JokerSamples
from tqdm import tqdm
import yaml
from schwimmbad import SerialPool
from schwimmbad.mpi import MPIAsyncPool

# Project
from hq.data import get_rvdata
from hq.log import logger
from hq.plot import plot_mcmc_diagnostic
from hq.config import (HQ_CACHE_PATH, config_to_jokerparams, config_to_alldata)
from hq.script_helpers import get_parser


def worker(joker, apogee_id, data, config, MAP_sample, emcee_cache_path,
           overwrite):
    chain_file = join(emcee_cache_path, '{0}.npz'.format(apogee_id))
    plot_file = join(emcee_cache_path, '{0}.png'.format(apogee_id))
    model_file = join(emcee_cache_path, 'model.pickle')

    emcee_sampler = None
    if not exists(chain_file) or overwrite:
        t0 = time.time()
        logger.log(1, "{0}: Starting emcee sampling".format(apogee_id))

        try:
            model, samples, emcee_sampler = joker.mcmc_sample(
                data, MAP_sample,
                n_burn=config['emcee']['n_burn'],
                n_steps=config['emcee']['n_steps'],
                n_walkers=config['emcee']['n_walkers'],
                return_sampler=True)
        except Exception as e:
            logger.warning("\t Failed emcee sampling for star {0} \n Error: {1}"
                           .format(apogee_id, str(e)))
            return None

        logger.debug("{0}: done sampling - {1} raw samples returned "
                     "({2:.2f} seconds)".format(apogee_id, len(samples),
                                                time.time() - t0))

        np.savez(chain_file, (emcee_sampler.chain, emcee_sampler.lnprobability))

        if not exists(model_file):
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

    if not exists(plot_file):
        logger.debug('Making plots for {0}'.format(apogee_id))

        if emcee_sampler is None:
            samples = np.load(chain_file)
            chain = samples['arr_0'][0]

        fig = plot_mcmc_diagnostic(chain)
        fig.savefig(plot_file, dpi=250)
        plt.close(fig)


def main(run_name, pool, overwrite=False):
    run_path = join(HQ_CACHE_PATH, run_name)
    with open(join(run_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())

    emcee_cache_path = join(HQ_CACHE_PATH, run_name, 'emcee')
    os.makedirs(emcee_cache_path, exist_ok=True)

    # Get paths to files needed to run
    params = config_to_jokerparams(config)
    joker = TheJoker(params)

    # Load the analyzed joker samplings file, only keep unimodal:
    joker_metadata = QTable.read(join(HQ_CACHE_PATH, run_name,
                                      '{0}-metadata.fits'.format(run_name)))
    unimodal_tbl = joker_metadata[joker_metadata['unimodal']]

    # Load the data:
    allstar, allvisit = config_to_alldata(config)
    allstar = allstar[np.isin(allstar['APOGEE_ID'], unimodal_tbl['APOGEE_ID'])]
    allvisit = allvisit[np.isin(allvisit['APOGEE_ID'], allstar['APOGEE_ID'])]

    tasks = []
    logger.debug("Loading data and preparing tasks...")
    for row in tqdm(unimodal_tbl):
        visits = allvisit[allvisit['APOGEE_ID'] == row['APOGEE_ID']]
        data = get_rvdata(visits)

        # Load the MAP sample:
        vals = dict([(k[4:], row[k])
                     for k in row.colnames
                     if k.startswith('MAP_') and not k.startswith('MAP_ln')])

        MAP_sample = JokerSamples(**vals)
        MAP_sample.t0 = data.t0

        tasks.append([joker, row['APOGEE_ID'], data, config, MAP_sample,
                      emcee_cache_path, overwrite])

    logger.info('Done preparing tasks: {0} stars in process queue'
                .format(len(tasks)))

    for r in tqdm(pool.starmap(worker, tasks), total=len(tasks)):
        pass


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='Run The Joker on APOGEE data',
                        loggers=[logger, joker_logger])

    parser.add_argument("--name", dest="run_name", required=True,
                        type=str, help="The name of the run.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true",
                        help="Overwrite any existing samplings")

    args = parser.parse_args()

    if args.mpi:
        Pool = MPIAsyncPool
    else:
        Pool = SerialPool

    with Pool() as pool:
        main(run_name=args.run_name, pool=pool, overwrite=args.overwrite)

    sys.exit(0)
