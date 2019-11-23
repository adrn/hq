# Standard library
import os
import pymc3 as pm
import sys
from hq.script_helpers import get_parser
import theano
theano.config.optimizer = 'None'


def worker(task):
    i, pars = task
    samples = pm.distributions.draw_values(pars, size=10_000)
    print(i)
    return samples


def main(pool):
    import theano
    theano.config.optimizer = 'None'

    with pm.Model() as model:
        par1 = pm.Normal('a', 0, 5)
        par2 = pm.Normal('b', 5, 2)
        par3 = pm.Deterministic('derp', par1 + par2)

    pars = [par1, par2, par3]
    tasks = [(i, pars) for i in range(100)]

    all_samples = []
    for samples in pool.map(worker, tasks):
        if samples is not None:
            all_samples.append(samples)

    print(len(all_samples))
    print(all_samples[0])


if __name__ == "__main__":
    # Define parser object
    parser = get_parser(description='Generate an HQ run template with the '
                                    'specified name. After running this, you '
                                    'then have to go in to the run directory '
                                    'and edit the configuration.',
                        loggers=[])

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()

    if args.mpi:
        from schwimmbad.mpi import MPIAsyncPool
        Pool = MPIAsyncPool
        kw = dict()
    elif args.n_procs > 1:
        from schwimmbad import MultiPool
        Pool = MultiPool
        kw = dict(processes=args.n_procs)
    else:
        from schwimmbad import SerialPool
        Pool = SerialPool
        kw = dict()

    args = parser.parse_args()

    with Pool(**kw) as pool:
        main(pool)

    sys.exit(0)
