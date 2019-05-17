# TODO: is this still needed?

# Third-party
import h5py
from thejoker.sampler import JokerSamples
from thejoker.utils import quantity_from_hdf5

__all__ = ['load_samples']


def load_samples(group_or_filename, apogee_id=None, return_logprobs=False,
                 **kwargs):
    """A wrapper around `thejoker.JokerSamples.from_hdf5` that...TODO

    Parameters
    ----------
    group_or_filename : :class:`h5py.Group` or str
    apogee_id : str, optional
        If a filename is passed to ``group_or_filename``, you must also specify
        the APOGEE ID of the source to load.
    """

    if isinstance(group_or_filename, str):
        if apogee_id is None:
            raise ValueError("If a filename is passed, you must also specify "
                             "the APOGEE ID of the source to load.")

        f = h5py.File(group_or_filename, 'r')
        group = f[apogee_id]

    else:
        f = None
        group = group_or_filename

    ln_prior = None
    ln_likelihood = None

    samples_dict = dict()
    for k in group.keys():
        if k == 'ln_prior': # skip prob values
            ln_prior = group['ln_prior']

        elif k == 'ln_likelihood': # skip prob values
            ln_likelihood = group['ln_likelihood']

        else:
            samples_dict[k] = quantity_from_hdf5(group, k)

    if f is not None:
        f.close()

    kwargs.update(samples_dict)

    if return_logprobs:
        return JokerSamples(**kwargs), ln_prior, ln_likelihood
    else:
        return JokerSamples(**kwargs)
