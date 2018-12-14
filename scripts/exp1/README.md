Experiment 1: beliefs about jitter in DR13 vs. DR14
===================================================

Here we use The Joker to sample over orbital parameters for batches of stars
with 4, 8, and 16 visits that appear in both DR13 and DR14.
For the most recent run, we fix the jitter to the logg-dependent value from
Troup et al. 2016 (actually from Hekker et al. 2008):
jitter = 2(0.015)^{1/3 logg} km/s

I previously ran this while sampling in the jitter - I saved those cache files
in cache/sample-jitter.
