# Imports we need for defining the prior:
import astropy.units as u
import pymc as pm
import thejoker as tj
import thejoker.units as xu

# The prior used to run The Joker:
with pm.Model() as model:
    # Prior on the extra variance parameter:
    s = xu.with_unit(pm.Lognormal("s", 0, 0.5), u.km / u.s)

    # Set up the default Joker prior:
    prior = tj.JokerPrior.default(
        P_min=2 * u.day,
        P_max=1024 * u.day,
        sigma_K0=30 * u.km / u.s,
        sigma_v=100 * u.km / u.s,
        s=s,
    )
