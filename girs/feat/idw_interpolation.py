import numpy as np
from scipy import spatial


def idw(xy_in, z_in, xy_out, nnn=8, p=2):
    """
    :param xy_in: list of coordinate tuples
    :param z_in: list of: 1) values or 2) list of values => len(xy_in==len(z_in))
    :param xy_out: list of coordinate tuples
    :param nnn: integer, number of nearest neighbors
    :return: z value or list of z values
    """
    assert len(xy_in) == len(z_in)

    z_in = np.asarray(z_in)
    nnn = min(nnn, len(xy_in))
    distances, indices = spatial.KDTree(xy_in).query(xy_out, k=nnn)
    z_out = np.ndarray(shape=(len(indices), len(z_in[0])), dtype=float)
    z_out.fill(0.0)
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if nnn == 1:
            wz = z_in[idx]
        elif dist[0] < 1e-10:  # distances are sorted (nearest first)
            wz = z_in[idx[0]]
        else:
            w = 1.0 / dist**p
            w /= np.sum(w)
            wz = np.dot(w, z_in[idx])
        z_out[i] = wz

    return z_out if len(xy_out) > 1 else z_out[0]



