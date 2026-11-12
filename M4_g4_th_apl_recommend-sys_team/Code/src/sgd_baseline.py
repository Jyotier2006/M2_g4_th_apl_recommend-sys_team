"""Original M2 SGD kernel, unchanged; procedural wrapper is in experiments.py."""
import numpy as np
from numba import njit

@njit(cache=True)
def sgd_epoch(u, i, r, order, um, im, ub, ib, offset, lr, reg):
    for row in order:
        a, b = u[row], i[row]
        error = r[row] - offset - ub[a] - ib[b] - np.dot(um[a], im[b])
        ub[a] += lr * (error - reg * ub[a])
        ib[b] += lr * (error - reg * ib[b])
        for k in range(um.shape[1]):
            old_u, old_i = um[a, k], im[b, k]
            um[a, k] += lr * (error * old_i - reg * old_u)
            im[b, k] += lr * (error * old_u - reg * old_i)


