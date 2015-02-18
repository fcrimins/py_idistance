import numpy as np
cimport numpy as np
import heapq
from libc.math cimport sqrt # http://docs.cython.org/src/tutorial/external.html


def knn_search_sequential(dat, query_pt, int K_):
    """Search sequentially through every point in dat for query_pt's K_ nearest
    neighbors.
    """
    cdef double distj
    knn_heap = []
    for i, mat in enumerate(dat):
        sqdists = np.sum((mat - query_pt)**2, axis=1)
        for j in xrange(mat.shape[0]): # for each row/point
            distj = sqrt(sqdists[j])
            _add_neighbor(knn_heap, K_, (None, i, j), distj)
    return knn_heap            


# @TODO: use cpdef here? "The directive cpdef makes two versions of the method available; one fast for use from Cython and one slower for use from Python."
_neighbors_visited = 0
cdef void _add_neighbor(knn_heap, int K_, node, double dist_node):
    """Maintain a heap of the K_ closest neighbors
    """
    globals()['_neighbors_visited'] += 1
    # heapq maintains a "min heap" so we store by -dist
    heap_node = (-dist_node, node[1], node[2])
    #print('_add_neighbor: {}'.format(heap_node))
    
    # @TODO: only add neighbor if it isn't in the same cross validation bucket as the query point
    if len(knn_heap) < K_:
        heapq.heappush(knn_heap, heap_node) # @TODO: does this perform an assignment?  does it matter?
    
    # -knn_heap[0][0] is the distance to the farthest point in the current knn
    elif dist_node < -knn_heap[0][0]:
        heapq.heappushpop(knn_heap, heap_node)
