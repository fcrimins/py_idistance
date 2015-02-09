import numpy as np
import bisect
import heapq


def bplus_tree(dat):
    
    C_, maxs, mins = _calculate_c(dat)
    
    ref_pts = _reference_points(maxs, mins)
    
    idists, partition_dist_max = _idistance_index(dat, ref_pts, C_)
    
    query_pt = np.array([0.0, 0.0])
    query_pt = dat[0][0]
    K_ = 5
    print('KNN SEARCH')
    knn = _knn_search(dat, query_pt, K_, C_, ref_pts, idists, partition_dist_max)
    
    print('KNN SEARCH SEQUENTIAL')
#     knn_seq = _knn_search_sequential(dat, query_pt, K_)
    
    return 0


def _knn_search_sequential(dat, query_pt, K_):
    
    knn_heap = []
    
    for i, mat in enumerate(dat):
        
        dists = np.sqrt(np.sum((mat - query_pt)**2, axis=1))
        
        for j in xrange(mat.shape[0]): # for each row/point
            
            _add_neighbor(knn_heap, K_, (None, i, j), dists[j])
            
    return knn_heap            


def _knn_search(dat, query_pt, K_, C_, ref_pts, idists, partition_dist_max):

    radius = 0.01 # R_ (initial radius to search)
    
    knn_heap = []
    
    # variable to mark for partitions checked
    partition_checked = [False] * len(ref_pts)

    # arrays of iterators
    left_idxs = [None] * len(ref_pts)
    right_idxs = [None] * len(ref_pts)
    
    while len(knn_heap) < K_ and radius < C_:
        radius *= 2.0
        print('RADIUS = {}'.format(radius))
        _knn_search_radius(K_, knn_heap, dat, query_pt, radius, C_, ref_pts, left_idxs, right_idxs, partition_checked, idists, partition_dist_max)

    return knn_heap


def _knn_search_radius(K_, knn_heap, dat, query_pt, R_, C_, ref_pts, left_idxs, right_idxs, partition_checked, idists, partition_dist_max):

    for i, rp in enumerate(ref_pts):
    
        # calc distance from query point to partition i
        d_rp = np.sqrt(np.sum((rp - query_pt)**2))
        
        # calculate the iDistance/index of the query point (for the current ref point)
        q_idist = i * C_ + d_rp # @TODO: roundOff necessary?

        if not partition_checked[i]:
            
            # filter dist(O_i, q) - querydist(q)="r" <= dist_max_i
            # (if the search radius from q overlaps w/ the partition's max radius)
            if d_rp - R_ <= partition_dist_max[i]:
                
                partition_checked[i] = True
                
                # if query point is inside this partition, must search left and right
                if d_rp <= partition_dist_max[i]:
                    
                    # find query pt and search inwards/left and outwards/right
                    right_idxs[i] = bisect.bisect_right(idists, (q_idist, None, None)) # strictly greater than
                    left_idxs[i] = right_idxs[i] - 1 # <=
                    _knn_search_inward(K_, knn_heap, dat, idists, left_idxs, C_, q_idist - R_, query_pt, i)
                    _knn_search_outward(K_, knn_heap, dat, idists, right_idxs, C_, q_idist + R_, query_pt, i, partition_dist_max)

                else: # query sphere intersects, so only search inward towards the ref point
                    left_idxs[i] = bisect.bisect_right(idists, ((i + 1) * C_, None, None)) - 1 # <=
                    _knn_search_inward(K_, knn_heap, dat, idists, left_idxs, C_, q_idist - R_, query_pt, i)
                    right_idxs[i] = None
                    
        else: # we've checked this partition before
            if left_idxs[i] is not None:
                _knn_search_inward(K_, knn_heap, dat, idists, left_idxs, C_, q_idist - R_, query_pt, i)
            if right_idxs[i] is not None:
                _knn_search_outward(K_, knn_heap, dat, idists, right_idxs, C_, q_idist + R_, query_pt, i, partition_dist_max)


def _knn_search_inward(K_, knn_heap, dat, idists, left_idxs, C_, stopping_val, query_pt, part_i):
    """Iterate left_idxs[part_i] inward towards reference point until stopping value has
    been reached or partion has been exited.
    
    Parameters
    ----------
    left_idxs[part_i] : int
      Index into idists to the node to start searching from.
    stopping_val : double
      Stopping value (i.e. query index - radius).
    """
    partition_offset = part_i * C_ # lower partition boundary (b/c iterating down)
    
    node = idists[left_idxs[part_i]]
    print('Searching inward from {} ({})'.format(node, left_idxs[part_i]))

    # while not to stopping value and still inside partition
    while left_idxs[part_i] >= 0 and node[0] >= stopping_val and node[0] >= partition_offset:
        dist_node = np.sqrt(np.sum((dat[node[1]][node[2]] - query_pt)**2))
        _add_neighbor(knn_heap, K_, node, dist_node)
        left_idxs[part_i] -= 1
        if left_idxs[part_i] >= 0:
            node = idists[left_idxs[part_i]]
            
    # exited partition (i.e. reached reference point)
    if left_idxs[part_i] < 0 or node[0] < partition_offset: 
        left_idxs[part_i] = None


def _knn_search_outward(K_, knn_heap, dat, idists, right_idxs, C_, stopping_val, query_pt, part_i, partition_dist_max):
    """Iterate right_idxs[part_i] outward away from reference point until stopping value has
    been reached or partion has been exited.
    
    Parameters
    ----------
    left_idxs[part_i] : int
      Index into idists to the node to start searching from.
    stopping_val : double
      Stopping value (i.e. query index - radius).
    """
    idist_max = part_i * C_ + partition_dist_max[part_i]
    
    node = idists[right_idxs[part_i]]
    print('Searching outward from {} ({})'.format(node, right_idxs[part_i]))
    
    num_idists = len(idists)

    # while not to stopping value and still inside partition
    while right_idxs[part_i] < num_idists and node[0] <= stopping_val and node[0] <= idist_max:
        dist_node = np.sqrt(np.sum((dat[node[1]][node[2]] - query_pt)**2))
        _add_neighbor(knn_heap, K_, node, dist_node)
        right_idxs[part_i] += 1
        if right_idxs[part_i] < num_idists:
            node = idists[right_idxs[part_i]]

    # exited partition (i.e. reached ref point's hypersphere boundary)
    if right_idxs[part_i] >= num_idists or node[0] > idist_max: 
        right_idxs[part_i] = None


def _add_neighbor(knn_heap, K_, node, dist_node):
    """Maintain a heap of the K_ closest neighbors
    """
    # heapq maintains a "min heap" so we store by -dist
    heap_node = (-dist_node, node[1], node[2])
    
    print('_add_neighbor: {}'.format(heap_node))
    
    # @TODO: only add neighbor if it isn't in the same cross validation bucket as the query point
    if len(knn_heap) < K_:
        heapq.heappush(knn_heap, heap_node)
    else:
        heapq.heappushpop(knn_heap, heap_node)


def _idistance_index(dat, ref_pts, C_):
    """Compute nearest reference point to each point along with farthest point from
    each reference point.  Returns the "iDistance" for each point with reference point
    r_i, i.e. iDistance = i * C * d_ij, where C is the constant C and d is the distance
    from r_i to point j.  Also returns the distance to each reference point's farthest
    point along with its index.
    """
    idists = [] # @TODO: not sure this variable is necessary 
    sorted_idists = []
    partition_dist_max = [0.0] * len(ref_pts)
    partition_dist_max_idx = [None] * len(ref_pts) # @TODO: not sure this variable is necessary
    
    for i, mat in enumerate(dat): # for each data file matrix
        idists.append([])

        for j in xrange(mat.shape[0]): # for each row/point
            
            row = mat[j,:] # @TODO: bind this (and the following) method before the loops
            
            sq = (ref_pts - row)**2 # difference (to each ref point) per dim squared
            
            ssq = np.sum(sq, axis=1) # SSD to each ref point
            
            minr = np.argmin(ssq) # index of closest ref point
            ref_dist = np.sqrt(ssq[minr])
            
            # @TODO: is roundOff needed like it is in the C++ code?
            idist = minr * C_ + ref_dist
            idists[-1].append(idist)
            sorted_idists.append((idist, i, j))
            
            if ref_dist > partition_dist_max[minr]:
                partition_dist_max[minr] = ref_dist
                partition_dist_max_idx[minr] = (i, j)
        
    # convert idists into a sortable array so that we can perform binary search for
    # a query point with its iDistance rather than searching a B+ tree (given 4e6
    # points the former will be quicker, log2(4e6)=22 comparisons, compared to the
    # latter, 2*50=100 comparisons given 50 dimensions)
    sorted_idists.sort()

    return sorted_idists, partition_dist_max


def _reference_points(mins, maxs):
    """Choose reference points.  These will dictate how the metric space is
    partitioned, something along the lines of a Voronoi diagram.  Every other
    point will be assigned to its closest reference point.  Here we choose
    reference points for the midpoint of each dimension's range leaving all
    other dimensions at either their min or max (may want to change this to
    5% or 95% eventually as min/max may be too far outside the cloud thus
    creating excess overlap in the reference points' spheres).  May also want
    to add a center point, though this point could end up holding the majority
    of the points and mess with the performance of the algorithm, which could
    mean that randomly selected points might just be better as they will
    be distributed the same as the rest of the data set.
    """
    dim = len(mins)
    num_partitions = dim * 2 # may want to add a center point also
    mids = (mins + maxs) / 2
    ref_pts = []
    for i in xrange(num_partitions):
        ref_pts.append(np.copy(mids)) # need to ensure not using references to the same 'mids' ndarray
    
    for i in xrange(dim):
        ref_pts[i * 2    ][i] = mins[i]
        ref_pts[i * 2 + 1][i] = maxs[i]
        
    return ref_pts


def _calculate_c(dat):
    """Compute range r_i for each dimension and return the length of the diagonal
    (rounded up to the nearest power of 10) across all of the dimensions.  This
    is guaranteed to be longer than the distance between any two points.
    
    This constant will be used to "separate" points' distances to their reference
    points so that all points with a common closest reference point are grouped
    together in the eventual B+ tree.  For each reference point r_i where i in
    [0,n), a non-reference point x will be assigned an "iDistance" of i*c + d
    where d is the distance from x to its nearest reference point.  Since c is
    larger than all possible d's there won't be any overlap.
    """  
    mins = dat[0].min(axis=0)
    maxs = dat[0].max(axis=0)
    for mat in dat[1:]:
        mins = np.fmin(mins, mat.min(axis=0))
        maxs = np.fmax(maxs, mat.max(axis=0))
        
    ranges = maxs - mins

    # compute the diagonal    
    c = np.sqrt(sum(r**2 for r in ranges))
    logc = np.log10(c)
    c = 10**np.ceil(logc)
    return c, maxs, mins