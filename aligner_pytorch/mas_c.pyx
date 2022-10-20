
from cython.parallel import prange

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_mas_values(float[:, ::1] values, int m, int n) nogil:
    cdef int i
    cdef int j

    for j in range(1, n):
        values[0, j] = values[0, j-1] + values[0, j]

    for i in range(1, m):
        for j in range(i, n):
            values[i, j] = values[i, j] + max(values[i-1, j-1], values[i, j-1])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_mas_path(int[:, ::1] path, float[:, ::1] values, int m, int n) nogil:
    cdef int i = m - 1
    cdef int j

    for j in reversed(range(n)):
        path[i, j] = 1
        if i != 0 and (i == j or values[i, j-1] < values[i-1, j-1]):
            i -= 1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void compute_mas_alignement(int[:,:,::1] paths, float[:,:,::1] values, int[::1] ms, int[::1] ns) nogil:
    cdef int b = values.shape[0]
    cdef int i

    for i in prange(b, nogil=True):
        compute_mas_values(values[i], ms[i], ns[i])
        compute_mas_path(paths[i], values[i], ms[i], ns[i])
