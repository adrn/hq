# Imports we typically need for defining the prior:
import astropy.units as u
import pymc3 as pm
import exoplanet.units as xu
import thejoker as tj

# The prior used to run The Joker: please edit below
with pm.Model() as model:
    # Set up the default Joker prior:
    prior = tj.JokerPrior.default(
        P_min=None, P_max=None,
        sigma_K0=None,
        sigma_v=None
    )