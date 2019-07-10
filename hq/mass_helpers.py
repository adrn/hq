import numpy as np

# For abundance transformations:
log_eC = 8.39
log_eN = 7.78

# Upper-triangle matrix from Martig et al.
M = np.array([[95.87, -10.4, 41.36, 15.05, -67.61, -144.18, -9.42],
              [0., -0.73, -5.32, -0.93, 7.05, 5.12, 1.52],
              [0., 0., -46.78, -30.52, 133.58, -73.77, 16.04],
              [0., 0., 0., -1.61, 38.94, -15.29, 1.35],
              [0., 0., 0., 0., -88.99, 101.75, -18.65],
              [0., 0., 0., 0., 0., 27.77, 28.8],
              [0., 0., 0., 0., 0., 0., -4.1]])


def CM_NM_to_CNM(M_H, C_M, N_M):
    """Compute [(C+N)/M] using Keith's method.
    Parameters
    ----------
    M_H : numeric
        [M/H], metallicity.
    C_M : numeric
        [C/M], carbon abundance.
    N_M : numeric
        [N/M], nitrogen abundance.
    """

    C_H = C_M + M_H
    N_H = N_M + M_H

    N_C = 10**(C_H + log_eC)
    N_N = 10**(N_H + log_eN)
    N_CN = 10**log_eC + 10**log_eN

    CN_H = np.log10(N_C + N_N) - np.log10(N_CN)
    return CN_H - M_H


def get_martig_mask(data):
    # Teff, logg, vmicro, M_H, C_M, N_M
    fparam = data['FPARAM']

    CN = fparam[:, 4] - fparam[:, 5]
    mask = ((fparam[:, 3] > -0.8) &
            (fparam[:, 0] > 4000) & (fparam[:, 0] < 5000) &
            (fparam[:, 1] > 1.8) & (fparam[:, 1] < 3.3) &
            (fparam[:, 4] > -0.25) & (fparam[:, 4] < 0.15) &
            (fparam[:, 5] > -0.1) & (fparam[:, 5] < 0.45) &
            (CN > -0.6) & (CN < 0.2))

    return mask


def get_martig_vec(Teff, logg, M_H, C_M, N_M):
    """Produces a 1D vector that can be inner-producted with the upper-triangle
    matrix provided by Martig et al. to estimate the stellar mass.
    """
    Teff = np.array(Teff)

    if Teff.shape:
        vec = np.ones((7, Teff.shape[0]))
    else:
        vec = np.ones((7, ))

    vec[1] = M_H
    vec[2] = C_M
    vec[3] = N_M
    vec[4] = CM_NM_to_CNM(M_H, C_M, N_M)
    vec[5] = Teff / 4000.
    vec[6] = logg

    return vec.T


def get_martig_mass(allstar):
    ms = np.full(np.nan, len(allstar))
    mask = get_martig_mask(allstar)

    # subset of allStar rows that are within the martig selection
    martig_t = allstar[mask]
    vec = get_martig_vec(martig_t['FPARAM'][:, 0], martig_t['FPARAM'][:, 1],
                         martig_t['FPARAM'][:, 3], martig_t['FPARAM'][:, 4],
                         martig_t['FPARAM'][:, 5])
    ms[mask] = np.einsum('ij,ni,nj->n', M, vec, vec)

    return ms
