# Standard library
import os
import sys
import time

# Third-party
import h5py
import numpy as np

# Project
from hq.log import logger


def worker(task):

    for i in range(1_000_000):
        j = i**2

    return {'j': j}


def callback(result):
    if result is None:
        return


def main(pool):
    tasks = np.arange(1_000)

    logger.info(f'Done preparing tasks: split into {len(tasks)} task chunks')
    for r in pool.map(worker, tasks, callback=callback):
        pass


if __name__ == '__main__':
    from schwimmbad.mpi import MPIPool

    with MPIPool() as pool:
        main(pool=pool)

    sys.exit(0)
