import numpy as np
cimport numpy as np
import heapq
from libc.math cimport sqrt # http://docs.cython.org/src/tutorial/external.html
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double euclidean_distance(double[::1] x1,
                               double[::1] x2):
    """https://jakevdp.github.io/blog/2012/08/08/memoryview-benchmarks/"""
    cdef double tmp, d
    cdef np.intp_t i, N # why not just use int? https://github.com/scikit-learn/scikit-learn/pull/1458

    d = 0
    N = x1.shape[0]
    # assume x2 has the same shape as x1.  This could be dangerous!

    for i in range(N):
        tmp = x1[i] - x2[i]
        d += tmp * tmp

    return sqrt(d)


def knn_search_sequential(dat, query_pt, int K_):
    """Search sequentially through every point in dat for query_pt's K_ nearest
    neighbors.
    """
    cdef double distj
    cdef np.intp_t i, j
    cdef double[:, ::1] X
    
    knn_heap = []
    for i, mat in enumerate(dat):
        
        X = mat
        
        for j in xrange(X.shape[0]): # for each row/point
            distj = euclidean_distance(query_pt, X[j])
            _add_neighbor(knn_heap, K_, i, j, distj)
            
    return knn_heap            


cdef struct heap_node_t:
    double neg_dist
    np.intp_t mat_idx
    np.intp_t row_idx

# @TODO: use cpdef here? "The directive cpdef makes two versions of the method available; one fast for use from Cython and one slower for use from Python."
_neighbors_visited = 0
cdef void _add_neighbor(knn_heap, int K_, np.intp_t mat_idx, np.intp_t row_idx, double dist_node):
    """Maintain a heap of the K_ closest neighbors
    """
    globals()['_neighbors_visited'] += 1
    # heapq maintains a "min heap" so we store by -dist
    cdef heap_node_t heap_node
    heap_node.neg_dist = -dist_node
    heap_node.mat_idx = mat_idx
    heap_node.row_idx
    #print('_add_neighbor: {}'.format(heap_node))
    
    # @TODO: only add neighbor if it isn't in the same cross validation bucket as the query point
    if len(knn_heap) < K_:
        heapq.heappush(knn_heap, heap_node) # @TODO: does this perform an assignment?  does it matter?
    
    # -knn_heap[0][0] is the distance to the farthest point in the current knn
    elif dist_node < -knn_heap[0][0]:
        heapq.heappushpop(knn_heap, heap_node)
