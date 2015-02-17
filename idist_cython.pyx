import numpy as np
import heapq


def knn_search_sequential(dat, query_pt, int K_):
    """Search sequentially through every point in dat for query_pt's K_ nearest
    neighbors.
    """
    knn_heap = []
    for i, mat in enumerate(dat):
        dists = np.sqrt(np.sum((mat - query_pt)**2, axis=1))
        for j in xrange(mat.shape[0]): # for each row/point
            _add_neighbor(knn_heap, K_, (None, i, j), dists[j])
    return knn_heap            

_neighbors_visited = 0
cdef _add_neighbor(knn_heap, int K_, node, dist_node):
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
