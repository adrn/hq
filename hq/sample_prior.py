import astropy.units as u
import h5py
import numpy as np
from thejoker.sampler import pack_prior_samples


def make_prior_cache(filename, thejoker, nsamples, batch_size=None):
    """

    Parameters
    ----------
    filename : str
        The HDF5 file name to cache to.
    thejoker : `~thejoker.sampler.TheJoker`
        An instance of ``TheJoker``.
    ntotal : int
        Number of samples to generate in the cache.
    batch_size : int (optional)
        The batch size to generate each iteration. Defaults to ``nsamples/512``.

    """
    nsamples = int(nsamples)

    if batch_size is None:
        batch_size = nsamples // 512
    batch_size = int(batch_size)

    # first just make an empty file
    with h5py.File(filename, 'w') as f:
        pass

    num_added = 0
    for i in range(2**16): # HACK: magic number, maximum num. iterations
        samples, ln_probs = thejoker.sample_prior(min(batch_size, nsamples),
                                                  return_logprobs=True)

        # TODO: make units configurable?
        packed_samples, units = pack_prior_samples(samples, u.km/u.s)

        size, K = packed_samples.shape

        if (num_added + size) > nsamples:
            packed_samples = packed_samples[:nsamples - (num_added + size)]
            size, K = packed_samples.shape

        if size <= 0:
            break

        with h5py.File(filename, 'r+') as f:
            if 'samples' not in f:
                # make the HDF5 file with placeholder datasets
                f.create_dataset('samples', shape=(nsamples, K),
                                 dtype=np.float32)
                f.create_dataset('ln_prior_probs', shape=(nsamples,),
                                 dtype=np.float32)
                f.attrs['units'] = np.array([str(x)
                                             for x in units]).astype('|S6')

            i1 = num_added
            i2 = num_added + size

            f['samples'][i1:i2, :] = packed_samples[:size]
            f['ln_prior_probs'][i1:i2] = ln_probs[:size]

        num_added += size

        if num_added >= nsamples:
            break
