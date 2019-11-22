import os
import h5py

__all__ = ['make_prior_cache']


def make_prior_cache(filename, prior, n_total_samples, batch_size=None):
    """

    Parameters
    ----------
    filename : str
        The HDF5 file name to cache to.
    prior : `~thejoker.JokerPrior`
        The prior.
    n_total_samples : int
        Number of samples to generate in the cache.
    batch_size : int (optional)
        The batch size to generate each iteration. Defaults to
        ``n_total_samples / 512``.

    """
    n_total_samples = int(n_total_samples)

    if batch_size is None:
        batch_size = n_total_samples // 512  # MAGIC NUMBER
    batch_size = int(batch_size)

    # first just make an empty file
    with h5py.File(filename, 'w'):
        pass

    num_added = 0
    for i in range(4096):  # HACK: magic number, maximum num. iterations
        samples = prior.sample(min(batch_size, n_total_samples),
                               return_logprobs=True)

        size = len(samples)

        if (num_added + size) > n_total_samples:
            samples = samples[:n_total_samples - (num_added + size)]

        if size <= 0:
            break

        samples.write(filename, append=os.path.exists(filename))

        num_added += size
        if num_added >= n_total_samples:
            break
