#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
import heapq
from libc.math cimport sqrt # http://docs.cython.org/src/tutorial/external.html
cimport cython

# can't include this, even if include_path passed to cythonize is set correctly
# because it requires many of the functions that are implemented in ball_tree.pxd
# like VALID_METRICS and allocate_data, so instead NeighborsHeap has merely
# been copied at the bottom 
#include "binary_tree.pxi"

# DOESN'T WORK AS THERE's NO typedefs.pxd FILE, i.e. source code is not included, i.e.
# these have been compiled into C, perhaps they'd be accessible with 'cdef extern'
#from sklearn.neighbors cimport typedefs # cimport DTYPE_t, ITYPE_t, DITYPE_t
ctypedef np.float64_t DTYPE_t  # WARNING: should match DTYPE in typedefs.pyx
ctypedef np.intp_t ITYPE_t  # WARNING: should match ITYPE in typedefs.pyx
from sklearn.neighbors.typedefs import DTYPE, ITYPE # this works b/c these are Python types (used in NeighborsHeap.__cinit__)

from cython.parallel cimport prange
cimport openmp


#from sklearn.neighbors import dist_metrics # works, just not useful
#from sklearn.neighbors.dist_metrics cimport euclidean_rdist import # "Name 'euclidean_rdist' not declared in module..." 
cdef inline DTYPE_t euclidean_rdist(DTYPE_t * x1, DTYPE_t * x2,
                                    ITYPE_t size) nogil except -1:
    """Copied from sklearn.neighbors.dist_metrics.pxd."""
    cdef DTYPE_t tmp, d=0
    cdef np.intp_t j
    for j in range(size): # @TODO: range faster?
        tmp = x1[j] - x2[j]
        d += tmp * tmp
    return d

_ndists2 = 0

cpdef num_procs():
    return openmp.omp_get_num_procs()
 
def knn_search_sequential(dat, query_pt, int K_):
    """Search sequentially through every point in dat for query_pt's K_ nearest
    neighbors.
    Also: https://jakevdp.github.io/blog/2012/08/08/memoryview-benchmarks/
    """
    cdef np.intp_t i, j, k, n, nc, rc, chunk, jk
    cdef DTYPE_t[:, ::1] X_
    cdef DTYPE_t sqdistj
    cdef ITYPE_t D_ = dat[0].shape[1]
    cdef DTYPE_t[::1] Q_ = query_pt
    
    if len(dat) != 1:
        raise ValueError('len(dat) != 1 which is required now that we\'re using NeighborsHeap')

    cdef NeighborsHeap heap = NeighborsHeap(1, K_)
    for i, mat in enumerate(dat):

        X_ = mat # convert to memoryview so that we can take addresses below
        n = X_.shape[0]
        globals()['_ndists2'] += n

        # for i in prange(n, nogil=True):
        #     with gil:
        #         for j in range(100):
        #             x[i,:] = np.cos(x[i,:])

        chunk = 103
        nc = n / chunk
        rc = n % chunk

        with nogil, cython.boundscheck(False), cython.wraparound(False):
            for j in prange(nc, schedule='static', num_threads=2):
                for k in range(chunk):
                    jk = j * chunk + k
                    sqdistj = euclidean_rdist(&Q_[0], &X_[jk, 0], D_)
                    heap.push(<ITYPE_t>0, <DTYPE_t>sqdistj, <ITYPE_t>jk) # _add_neighbor
            for k in range(n % chunk):
                jk = nc * chunk + k
                sqdistj = euclidean_rdist(&Q_[0], &X_[jk, 0], D_)
                heap.push(<ITYPE_t>0, <DTYPE_t>sqdistj, <ITYPE_t>jk) # _add_neighbor

    # sqrt all of the reduced/squared distances to get Euclidean distances
    knn_heap = []
    for i in range(K_):
        knn_heap.append((-sqrt(heap.distances[0, i]), 0, heap.indices[0, i]))
 
    return knn_heap
 
 
 
##################################################################################
  
cdef inline void dual_swap(DTYPE_t* darr, ITYPE_t* iarr,
                           ITYPE_t i1, ITYPE_t i2):
    """swap the values at inex i1 and i2 of both darr and iarr"""
    cdef DTYPE_t dtmp = darr[i1]
    darr[i1] = darr[i2]
    darr[i2] = dtmp
  
    cdef ITYPE_t itmp = iarr[i1]
    iarr[i1] = iarr[i2]
    iarr[i2] = itmp
  
  
cdef class NeighborsHeap:
    """A max-heap structure to keep track of distances/indices of neighbors
  
    This implements an efficient pre-allocated set of fixed-size heaps
    for chasing neighbors, holding both an index and a distance.
    When any row of the heap is full, adding an additional point will push
    the furthest point off the heap.
  
    Parameters
    ----------
    n_pts : int
        the number of heaps to use
    n_nbrs : int
        the size of each heap.
    """
    cdef np.ndarray distances_arr
    cdef np.ndarray indices_arr
  
    cdef DTYPE_t[:, ::1] distances
    cdef ITYPE_t[:, ::1] indices
  
    def __cinit__(self):
        self.distances_arr = np.zeros((1, 1), dtype=DTYPE, order='C')
        self.indices_arr = np.zeros((1, 1), dtype=ITYPE, order='C')
        self.distances = get_memview_DTYPE_2D(self.distances_arr)
        self.indices = get_memview_ITYPE_2D(self.indices_arr)
  
    def __init__(self, n_pts, n_nbrs):
        self.distances_arr = np.inf + np.zeros((n_pts, n_nbrs), dtype=DTYPE,
                                               order='C')
        self.indices_arr = np.zeros((n_pts, n_nbrs), dtype=ITYPE, order='C')
        self.distances = get_memview_DTYPE_2D(self.distances_arr)
        self.indices = get_memview_ITYPE_2D(self.indices_arr)
  
    def get_arrays(self, sort=True):
        """Get the arrays of distances and indices within the heap.
  
        If sort=True, then simultaneously sort the indices and distances,
        so the closer points are listed first.
        """
        if sort:
            self._sort()
        return self.distances_arr, self.indices_arr
  
    cdef inline DTYPE_t largest(self, ITYPE_t row) except -1:
        """Return the largest distance in the given row"""
        return self.distances[row, 0]
  
    cdef int push(self, ITYPE_t row, DTYPE_t val, ITYPE_t i_val) nogil except -1:
        """push (val, i_val) into the given row"""
        cdef ITYPE_t i, ic1, ic2, i_swap
        cdef ITYPE_t size = self.distances.shape[1]
        cdef DTYPE_t* dist_arr = &self.distances[row, 0]
        cdef ITYPE_t* ind_arr = &self.indices[row, 0]
  
        # check if val should be in heap
        if val > dist_arr[0]:
            return 0
  
        # insert val at position zero
        dist_arr[0] = val
        ind_arr[0] = i_val
  
        #descend the heap, swapping values until the max heap criterion is met
        i = 0
        while True:
            ic1 = 2 * i + 1
            ic2 = ic1 + 1
  
            if ic1 >= size:
                break
            elif ic2 >= size:
                if dist_arr[ic1] > val:
                    i_swap = ic1
                else:
                    break
            elif dist_arr[ic1] >= dist_arr[ic2]:
                if val < dist_arr[ic1]:
                    i_swap = ic1
                else:
                    break
            else:
                if val < dist_arr[ic2]:
                    i_swap = ic2
                else:
                    break
  
            dist_arr[i] = dist_arr[i_swap]
            ind_arr[i] = ind_arr[i_swap]
  
            i = i_swap
  
        dist_arr[i] = val
        ind_arr[i] = i_val
  
        return 0
  
    cdef int _sort(self) except -1:
        """simultaneously sort the distances and indices"""
        cdef DTYPE_t[:, ::1] distances = self.distances
        cdef ITYPE_t[:, ::1] indices = self.indices
        cdef ITYPE_t row
        for row in range(distances.shape[0]):
            _simultaneous_sort(&distances[row, 0],
                               &indices[row, 0],
                               distances.shape[1])
        return 0
  
  
cdef int _simultaneous_sort(DTYPE_t* dist, ITYPE_t* idx,
                            ITYPE_t size) except -1:
    """
    Perform a recursive quicksort on the dist array, simultaneously
    performing the same swaps on the idx array.  The equivalent in
    numpy (though quite a bit slower) is
  
    def simultaneous_sort(dist, idx):
        i = np.argsort(dist)
        return dist[i], idx[i]
    """
    cdef ITYPE_t pivot_idx, i, store_idx
    cdef DTYPE_t pivot_val
  
    # in the small-array case, do things efficiently
    if size <= 1:
        pass
    elif size == 2:
        if dist[0] > dist[1]:
            dual_swap(dist, idx, 0, 1)
    elif size == 3:
        if dist[0] > dist[1]:
            dual_swap(dist, idx, 0, 1)
        if dist[1] > dist[2]:
            dual_swap(dist, idx, 1, 2)
            if dist[0] > dist[1]:
                dual_swap(dist, idx, 0, 1)
    else:
        # Determine the pivot using the median-of-three rule.
        # The smallest of the three is moved to the beginning of the array,
        # the middle (the pivot value) is moved to the end, and the largest
        # is moved to the pivot index.
        pivot_idx = size / 2
        if dist[0] > dist[size - 1]:
            dual_swap(dist, idx, 0, size - 1)
        if dist[size - 1] > dist[pivot_idx]:
            dual_swap(dist, idx, size - 1, pivot_idx)
            if dist[0] > dist[size - 1]:
                dual_swap(dist, idx, 0, size - 1)
        pivot_val = dist[size - 1]
  
        # partition indices about pivot.  At the end of this operation,
        # pivot_idx will contain the pivot value, everything to the left
        # will be smaller, and everything to the right will be larger.
        store_idx = 0
        for i in range(size - 1):
            if dist[i] < pivot_val:
                dual_swap(dist, idx, i, store_idx)
                store_idx += 1
        dual_swap(dist, idx, store_idx, size - 1)
        pivot_idx = store_idx
  
        # recursively sort each side of the pivot
        if pivot_idx > 1:
            _simultaneous_sort(dist, idx, pivot_idx)
        if pivot_idx + 2 < size:
            _simultaneous_sort(dist + pivot_idx + 1,
                               idx + pivot_idx + 1,
                               size - pivot_idx - 1)
    return 0

# Numpy 1.3-1.4 compatibility utilities
cdef DTYPE_t[:, ::1] get_memview_DTYPE_2D(
                               np.ndarray[DTYPE_t, ndim=2, mode='c'] X):
    return <DTYPE_t[:X.shape[0], :X.shape[1]:1]> (<DTYPE_t*> X.data)

cdef ITYPE_t[:, ::1] get_memview_ITYPE_2D(
                               np.ndarray[ITYPE_t, ndim=2, mode='c'] X):
    return <ITYPE_t[:X.shape[0], :X.shape[1]:1]> (<ITYPE_t*> X.data)

