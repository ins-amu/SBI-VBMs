import numpy as np
from numba import jit


def tril_indices_column(N, k=0):
    row_i, col_i = np.nonzero(
        np.tril(np.ones(N), k=k).T)  # Matlab works in column-major order, while Numpy works in row-major.
    Isubdiag = (col_i,
                row_i)  # Thus, I have to do this little trick: Transpose, generate the indices, and then "transpose" again...
    return Isubdiag


def build_matrix(linear_phfcd, size, k=-1):
    tri = np.zeros((size, size))
    i_lower = tril_indices_column(size, k=k)
    tri[i_lower] = linear_phfcd
    tri.T[i_lower] = tri[i_lower]  # make symmetric matrix
    return tri
