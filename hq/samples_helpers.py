from thejoker import JokerSamples

__all__ = ['read_samples_block']


def read_samples_block(c, tbl, **kwargs):
    """
    Create a JokerSamples instance from a section of a table of samples

    Parameters
    ----------
    c : `hq.config.Config`
    tbl : array_like or table_like

    Returns
    -------
    samples : `thejoker.JokerSamples`
    """

    prior = c.get_prior()

    samples = JokerSamples(poly_trend=prior.poly_trend, **kwargs)
    for key in prior.par_units:
        samples[key] = tbl[key] * prior.par_units[key]

    return samples
