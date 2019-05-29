# Standard library
from collections import defaultdict
from os.path import join, exists
import sys

# Third-party
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
import h5py
from tqdm import tqdm
import yaml
from thejoker import JokerSamples

# Project
from hq.data import get_rvdata
from hq.log import logger
from hq.config import (HQ_CACHE_PATH, config_to_alldata)
from hq.script_helpers import get_parser
from hq.samples_analysis import unimodal_P


def main(run_name):
    run_path = join(HQ_CACHE_PATH, run_name)
    with open(join(run_path, 'config.yml'), 'r') as f:
        config = yaml.load(f.read())

    # Get paths to files needed to run
    results_path = join(HQ_CACHE_PATH, run_name,
                        'thejoker-{0}.hdf5'.format(run_name))
    metadata_path = join(HQ_CACHE_PATH, run_name,
                         '{0}-metadata.fits'.format(run_name))

    # Load the data for this run:
    allstar, allvisit = config_to_alldata(config)

    n_requested_samples = config['requested_samples_per_star']
    poly_trend = config['hyperparams']['poly_trend']

    if not exists(results_path):
        raise IOError("Results file {0} does not exist! Did you run "
                      "run_apogee.py?".format(results_path))

    rows = defaultdict(list)
    with h5py.File(results_path, 'r') as results_f:
        for apogee_id in tqdm(results_f.keys(),
                              total=len(list(results_f.keys()))):
            # Load samples from The Joker and probabilities
            samples = JokerSamples.from_hdf5(results_f[apogee_id],
                                             poly_trend=poly_trend)
            ln_p = results_f[apogee_id]['ln_prior'][:]
            ln_l = results_f[apogee_id]['ln_likelihood'][:]

            # Load data
            visits = allvisit[allvisit['APOGEE_ID'] == apogee_id]
            data = get_rvdata(visits)

            rows['APOGEE_ID'].append(apogee_id)
            rows['n_visits'].append(len(data))

            MAP_idx = (ln_p + ln_l).argmax()
            MAP_sample = samples[MAP_idx:MAP_idx+1]
            for k in MAP_sample.keys():
                rows['MAP_'+k].append(MAP_sample[k][0])
            rows['t0'].append(MAP_sample.t0)

            if len(samples) == n_requested_samples:
                rows['joker_completed'].append(True)
            else:
                rows['joker_completed'].append(False)

            if unimodal_P(samples, data):
                rows['unimodal'].append(True)
            else:
                rows['unimodal'].append(False)

            rows['MAP_ln_likelihood'] = ln_l[MAP_idx]
            rows['MAP_ln_prior'] = ln_p[MAP_idx]

    for k in rows.keys():
        if isinstance(rows[k][0], (Time, u.Quantity)):
            rows[k] = rows[k][0].__class__(rows[k])

    tbl = Table(rows)
    tbl.write(metadata_path, overwrite=True)

    # TODO: read the metadata file with
    # Table.read(metadata_path, astropy_native=True)


if __name__ == '__main__':
    # Define parser object
    parser = get_parser(description='TODO',
                        loggers=logger)

    parser.add_argument("--name", dest="run_name", required=True,
                        type=str, help="The name of the run.")

    parser.add_argument("-o", "--overwrite", dest="overwrite", default=False,
                        action="store_true",
                        help="Overwrite any existing samplings")

    args = parser.parse_args()

    main(run_name=args.run_name)

    sys.exit(0)
