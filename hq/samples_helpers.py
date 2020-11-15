from thejoker import JokerSamples

__all__ = ['read_samples_block']


def read_samples_block(c, tbl):
    """Create a JokerSamples instance from a section of a pytables table of samples

    Parameters
    ----------
    c : `hq.config.Config`
    tbl : array_like or table_like

    Returns
    -------
    samples : `thejoker.JokerSamples`
    """

    prior = c.get_prior()

    units = prior._nonlinear_equiv_units.copy()
    units.update(prior._linear_equiv_units)

    samples = JokerSamples(poly_trend=prior.poly_trend)
    for key in units:
        samples[key] = tbl[key] * units[key]

    return samples
