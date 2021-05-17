# Imports we typically need for defining the prior:
import astropy.units as u
import pymc3 as pm
import exoplanet.units as xu
from exoplanet.distributions import Angle
import thejoker as tj

# TODO: update these to use get_prior() and get_prior_mcmc()

# The prior used to run The Joker: please edit below! Any parameters set to None
# should be changed to real values!
with pm.Model() as model:
    # Set up the default Joker prior:
    prior = tj.JokerPrior.default(
        P_min=None, P_max=None,
        sigma_K0=None,
        sigma_v=None
    )

with pm.Model() as model:
    # See note above: when running MCMC, we will sample in the parameters
    # (M0 - omega, omega) instead of (M0, omega)
    M0_m_omega = xu.with_unit(Angle('M0_m_omega'), u.radian)
    omega = xu.with_unit(Angle('omega'), u.radian)
    M0 = xu.with_unit(pm.Deterministic('M0', M0_m_omega + omega),
                      u.radian)

    prior_mcmc = tj.JokerPrior.default(
        P_min=None, P_max=None,
        sigma_K0=None,
        sigma_v=None,
        pars={'M0': M0, 'omega': omega}
    )


def custom_init_mcmc(**kwargs):
    """TODO: describe this"""
    pass