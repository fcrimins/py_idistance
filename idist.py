import logging
import numpy as np
import bisect
import heapq
import time
from sklearn.neighbors import BallTree, NearestNeighbors, KDTree

# http://docs.cython.org/src/tutorial/cython_tutorial.html#pyximport-cython-compilation-the-easy-way
# can't use pyximport b/c compiling idist_cython requires numpy include directory 
#import pyximport
#pyximport.install()
import idist_cython # not really idist but sequential search rather implemented in Cython

_DEBUG_ = False

_ndists = None

def bplus_tree(dat, iradius, K_):
    
    if K_ is None:
        K_ = 5

    C_, maxs, mins = _calculate_c(dat)
    
    print('C = {}'.format(C_))
    print('k = {}'.format(K_))
    print('D = {}'.format(dat[0].shape[1]))
    N_ = sum(mat.shape[0] for mat in dat)
    print('N = {}'.format(N_))
    
    if iradius is None:
        iradius = 0.2 # np.sqrt(C_* C_ / dat[0].shape[1]) * K_ / 400.0 # C_ / 50.0 e.g. 0.2, R_ (initial radius to search)
    #radius = C_ / num_points * np.power(K_, 1.0 / dat[0].shape[1]) * (iradius if iradius else 10.0)
    print('iradius = {}\n'.format(iradius))
    # @TODO: initalize radius to a fraction of C_; use some empirical tests to figure out optimal
    
    ref_pts = _reference_points(maxs, mins, dat)
    
    t0 = time.clock()
    idists, partition_dist_max, partition_point_counts = _idistance_index(dat, ref_pts, C_)
    time_index = time.clock() - t0
    kth_farthest_dists = []
    
    def partition_overlap(dat, ref_pts, partition_dist_max):
        max_iters = 1000
        niters = 0
        sum_nparts = 0
        for mat in dat:
            for i in xrange(mat.shape[0]):
                nparts = 0
                for j, rp in enumerate(ref_pts):
                    d_rp = np.sqrt(np.sum((rp - mat[i, :])**2))
                    if d_rp < partition_dist_max[j]:
                        nparts += 1
                niters += 1
                sum_nparts += nparts
                if niters >= max_iters:
                    break
            if niters >= max_iters:
                break
        print('avg number of partitions containing any given point: {:.4f} ({:2.0f}% of {} partitions)'.format(float(sum_nparts) / niters, float(sum_nparts) / niters * 100 / len(ref_pts), len(ref_pts)))
    
    
    assert(len(dat) == 1)
    t0 = time.clock()
    ball_tree = BallTree(dat[0], leaf_size=15)
    time_index_bt = time.clock() - t0
    
    # code: https://github.com/jakevdp/BinaryTree/blob/master/binary_tree.pxi (e.g. get_n_calls)
    OVERRIDE_BRUTE = True
    t0 = time.clock()
    if OVERRIDE_BRUTE:
        typ = BallTree
        brute_alg = 'override w/ {}'.format(typ) # BallTree_and_KDTree(N+1)=19 (per 1000 queries), KDTree(15)=70
        brute_tree = typ(dat[0], leaf_size=N_ + 1) # i.e. use BallTree w/ a huge leaf_size instead
    else:
        brute_alg = 'brute' # ball_tree=36.4, brute=68.5 (per 1000 queries)
        nbrs = NearestNeighbors(n_neighbors=K_, algorithm=brute_alg).fit(dat[0])
    print('brute_alg: {}\n'.format(brute_alg))
    time_index_brute = time.clock() - t0
    
    MAX_QUERIES = 10
    
    time_idist = 0.0
    time_seq = 0.0
    time_seq_cy = 0.0
    time_bt = 0.0
    time_brute = 0.0
    ndists_idist = 0
    ndists_seq = 0
    ndists_seq_cy = 0
    ndists_bt = 0
    ndists_brute = 0
    niters = 0

    for mat in dat:
        for j in xrange(mat.shape[0]):
            if niters > -1:
                globals()['stop_printing'] = None
            #query_pt = np.array([0.0, 0.0])
            #query_pt = dat[0][0]
            query_pt = mat[j,:]
            query_pt = np.copy(query_pt)
            #query_pt += [0.2, -0.1]
            
            t0 = time.clock()
            globals()['_ndists'] = 0
            #print('KNN SEARCH {}'.format(query_pt))
            knn = _knn_query_idist(dat, query_pt, K_, C_, ref_pts, idists, partition_dist_max, iradius, partition_point_counts)
            time_idist += time.clock() - t0
            ndists_idist += globals()['_ndists']
            
            t0 = time.clock()
            globals()['_ndists'] = 0
            knn_seq = _knn_query_sequential(dat, query_pt, K_)
            time_seq += time.clock() - t0
            ndists_seq += globals()['_ndists']
            
            t0 = time.clock()
            idist_cython._ndists2 = 0
            knn_seq_cy = idist_cython.knn_search_sequential(dat, query_pt, K_)
            time_seq_cy += time.clock() - t0
            ndists_seq_cy += idist_cython._ndists2
            
            ball_tree.reset_n_calls()
            t0 = time.clock()
            dist_bt, idx_bt = ball_tree.query(query_pt, k=K_, return_distance=True)
            time_bt += time.clock() - t0
            ndists_bt += ball_tree.get_n_calls() # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/binary_tree.pxi
            print('ball_tree(trims, leaves, splits) = {}'.format(ball_tree.get_tree_stats()))
            
            if OVERRIDE_BRUTE: brute_tree.reset_n_calls()
            t0 = time.clock()
            dist_brute, idx_brute = (brute_tree.query(query_pt, k=K_, return_distance=True) if OVERRIDE_BRUTE else
                                     nbrs.kneighbors(query_pt))
            time_brute += time.clock() - t0
            ndists_brute += (brute_tree.get_n_calls() if OVERRIDE_BRUTE else dat[0].shape[0])
            if OVERRIDE_BRUTE:
                print('brute_tree(trims, leaves, splits) = {}'.format(brute_tree.get_tree_stats()))
            
            def sk_2_knn(dist, idx):            
                knn = []
                for d, i in zip(dist[0,:], idx[0,:]):
                    knn.append((-d, 0, i))
                return knn
            
            knn_bt = sk_2_knn(dist_bt, idx_bt)
            knn_brute = sk_2_knn(dist_brute, idx_brute)
            
            knn.sort()
            knn_seq.sort()
            knn_seq_cy.sort()
            knn_bt.sort()
            knn_brute.sort()
            def neq_dists(knn0, knn1):
                return any(np.fabs(d0 - d1) > C_ / 1e6 for ((d0, _, _), (d1, _, _)) in zip(knn0, knn1))
            if neq_dists(knn_seq, knn):
                print('\niDistance KNN NOT EQUAL - {} (iter {})\n{}\n{}\n'.format(query_pt, niters, knn_seq, knn))
            if neq_dists(knn_seq, knn_seq_cy):
                print('\nCYTHON KNN NOT EQUAL - {} (iter {})\n{}\n{}\n'.format(query_pt, niters, knn_seq, knn_seq_cy))
            if neq_dists(knn_seq, knn_bt):
                print('\nBallTree KNN NOT EQUAL - {} (iter {})\n{}\n{}\n'.format(query_pt, niters, knn_seq, knn_bt))
            if neq_dists(knn_seq, knn_brute):
                print('\nBrute KNN NOT EQUAL - {} (iter {})\n{}\n{}\n'.format(query_pt, niters, knn_seq, knn_brute))
                
            print('k\'th ({}th) farthest distance: {:.2f}'.format(K_, -knn[0][0]))
            kth_farthest_dists.append(-knn[0][0])

            niters += 1
            if niters >= MAX_QUERIES:
                break
        if niters >= MAX_QUERIES:
            break
        
    # note correlated data in 50 dimensions is just like uncorrelated data in fewer dimensions
    # so since i'm using uncorrelated 50-dimensional data for testing, that's really like using
    # more dimensions than that given structured/correlated data
    # i.e. take a football cloud aligned along a dimension (that's uncorrelated) and spin it so that
    # it's diagonal (that's correlated)
    # now undo that rotation and fewer dimensions are required
    # this is the type of data that ball trees should help with b/c large parts of the space won't
    # need to be encircled
    # bottom line: 2 highly correlated dimensions really act like just a single dim (or maybe 1.1 dims)
        
    print('')
    partition_overlap(dat, ref_pts, partition_dist_max)
    print('partition_dist_max (min, max, avg): {:.2f}, {:.2f}, {:.2f}'.format(min(partition_dist_max), max(partition_dist_max), sum(partition_dist_max) / len(partition_dist_max)))
    print('kth_farthest_dists (min, max, avg): {:.2f}, {:.2f}, {:.2f}'.format(min(kth_farthest_dists), max(kth_farthest_dists), sum(kth_farthest_dists) / len(kth_farthest_dists)))
    print('*** kth_farthest / partition_dist_max: {:.3f} *** (iDistance breaks down when this approaches 1, which happens when the number of dimensions gets large)'.format(sum(kth_farthest_dists) / sum(partition_dist_max) / len(kth_farthest_dists) * len(partition_dist_max)))
    print('')
    print('* indexation time (idist, bt, brute): {:.4f}/{:.4f}/{:.4f}'.format(time_index, time_index_bt, time_index_brute))
    print('* absolute times per 1000 queries (seq, idist, seq_cy, bt, brute):  = {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}'.format(*(x * 1000 / niters for x in (time_seq, time_idist, time_seq_cy, time_bt, time_brute))))
    print('* sequential relative times (seq base (s) {:.4f}) (idist, seq_cy, bt, brute):  = {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(time_seq, time_idist/time_seq, time_seq_cy/time_seq, time_bt/time_seq, time_brute/time_seq))
    print('* neighbors (per iter, per N): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(float(ndists_seq) / niters / N_,
                                                               float(ndists_idist) / niters / N_,
                                                               float(ndists_seq_cy) / niters / N_,
                                                               float(ndists_bt) / niters / N_,
                                                               float(ndists_brute) / niters / N_))
    
    return 0


def _knn_query_sequential(dat, query_pt, K_):
    """Search sequentially through every point in dat for query_pt's K_ nearest
    neighbors.
    """
    knn_heap = []
    for i, mat in enumerate(dat):
        globals()['_ndists'] += mat.shape[0]
        sqdists = np.sum((mat - query_pt)**2, axis=1)
        for j in xrange(mat.shape[0]): # for each row/point
            _add_neighbor(knn_heap, K_, (None, i, j), sqdists[j])
    for i, nn in enumerate(knn_heap):
        knn_heap[i] = (-np.sqrt(-nn[0]), nn[1], nn[2])
    return knn_heap            


def _knn_query_idist(dat, query_pt, K_, C_, ref_pts, idists, partition_dist_max, iradius, partition_point_counts):
    
    num_points = sum(m.shape[0] for m in dat)

    radius = iradius    
    radius_increment = radius

    knn_heap = []
    
    # variable to mark for partitions checked
    partition_checked = [False] * len(ref_pts)

    # arrays of iterators
    left_idxs = [None] * len(ref_pts)
    right_idxs = [None] * len(ref_pts)
    
    # no need to calculate distance to every reference point every iteration
    globals()['_ndists'] += len(ref_pts)
    ref_pt_dists = []
    for i, rp in enumerate(ref_pts):
        
        # calculate distance from query point to partition i
        d_rp = np.sqrt(np.sum((rp - query_pt)**2))
        
        # calculate the iDistance/index of the query point (for the current ref point)
        ref_pt_dists.append((d_rp, i * C_ + d_rp))
    
    # -knn_heap[0][0] is the distance to the farthest point in the current knn, so as long
    # as radius is smaller than that, there could still be points outside of radius that are closer
    while radius < C_ and (len(knn_heap) < K_ or radius*radius < -knn_heap[0][0]):
        
        # no need to grow geometrically as search area is growing as the square of this already
        radius += radius_increment
        
        if _DEBUG_:
            print('RADIUS = {}'.format(radius))

        #print('RADIUS = {}'.format(radius))
        _knn_search_radius(K_, knn_heap, dat, query_pt, radius, C_, ref_pts, left_idxs, right_idxs, partition_checked, idists, partition_dist_max, ref_pt_dists)

    for i, nn in enumerate(knn_heap):
        knn_heap[i] = (-np.sqrt(-nn[0]), nn[1], nn[2])

    return knn_heap


def _knn_search_radius(K_, knn_heap, dat, query_pt, R_, C_, ref_pts, left_idxs, right_idxs, partition_checked, idists, partition_dist_max, ref_pt_dists):
    
    nparts_initialized = 0
    nparts_visited = 0

    for i, rp in enumerate(ref_pts):
        
        d_rp, q_idist = ref_pt_dists[i]
                
        if not partition_checked[i]:
            
            # filter dist(O_i, q) - querydist(q)="r" <= dist_max_i
            # (if the search radius from q overlaps w/ the partition's max radius)
            if d_rp - R_ <= partition_dist_max[i]:
                
                partition_checked[i] = True
                
                # if query point is inside this partition, must search left and right
                if d_rp <= partition_dist_max[i]:
                    
                    #print('initializing partition {} (INSIDE)'.format(i))
                    nparts_initialized += 1
                    
                    # find query pt and search inwards/left and outwards/right
                    right_idxs[i] = bisect.bisect_right(idists, (q_idist, -1, -1)) # strictly greater than
                    left_idxs[i] = right_idxs[i] - 1 # <=
                    _knn_search_inward(K_, knn_heap, dat, idists, left_idxs, C_, q_idist - R_, query_pt, i)
                    _knn_search_outward(K_, knn_heap, dat, idists, right_idxs, C_, q_idist + R_, query_pt, i, partition_dist_max)

                else: # query sphere intersects, so only search inward towards the ref point
                    #print('initializing partition {}'.format(i))
                    nparts_initialized += 1
                    left_idxs[i] = bisect.bisect_right(idists, ((i + 1) * C_, -1, -1)) - 1 # <=
                    _knn_search_inward(K_, knn_heap, dat, idists, left_idxs, C_, q_idist - R_, query_pt, i)
                    right_idxs[i] = None
                    
        else: # we've checked this partition before
            if left_idxs[i] is not None or right_idxs[i] is not None:
                nparts_visited += 1 
            if left_idxs[i] is not None:
                #print('checking partition {}'.format(i))
                _knn_search_inward(K_, knn_heap, dat, idists, left_idxs, C_, q_idist - R_, query_pt, i)
            if right_idxs[i] is not None:
                #print('checking partition {}'.format(i))
                _knn_search_outward(K_, knn_heap, dat, idists, right_idxs, C_, q_idist + R_, query_pt, i, partition_dist_max)
                
    if _DEBUG_:
        print('partitions initialized/visited: {}/{}'.format(nparts_initialized, nparts_visited))


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
    #print('Searching inward from {} ({})'.format(node, left_idxs[part_i]))
    search_start_idx = left_idxs[part_i]
    dists_visited = []

    # while not to stopping value and still inside partition
    while left_idxs[part_i] >= 0 and node[0] >= stopping_val and node[0] >= partition_offset:
        globals()['_ndists'] += 1
        sqdist_node = np.sum((dat[node[1]][node[2]] - query_pt)**2)
        new_neighbor = _add_neighbor(knn_heap, K_, node, sqdist_node)
        if _DEBUG_:
            d = float('{:.1f}'.format(np.sqrt(sqdist_node)))
            dists_visited.append((d, True) if new_neighbor else d)
        left_idxs[part_i] -= 1
        if left_idxs[part_i] >= 0:
            node = idists[left_idxs[part_i]]
            
    # exited partition (i.e. reached reference point)
    if left_idxs[part_i] < 0 or node[0] < partition_offset: 
        left_idxs[part_i] = None

    if dists_visited and _DEBUG_:
        print('    partition {} inward {} -> {} {}'.format(part_i, search_start_idx, left_idxs[part_i], dists_visited))

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
    #print('Searching outward from {} ({})'.format(node, right_idxs[part_i]))
    search_start_idx = right_idxs[part_i]
    dists_visited = []
    
    num_idists = len(idists)

    # while not to stopping value and still inside partition
    while right_idxs[part_i] < num_idists and node[0] <= stopping_val and node[0] <= idist_max:
        globals()['_ndists'] += 1
        sqdist_node = np.sum((dat[node[1]][node[2]] - query_pt)**2)
        new_neighbor = _add_neighbor(knn_heap, K_, node, sqdist_node)
        if _DEBUG_:
            d = float('{:.1f}'.format(np.sqrt(sqdist_node)))
            dists_visited.append((d, True) if new_neighbor else d)
        right_idxs[part_i] += 1
        if right_idxs[part_i] < num_idists:
            node = idists[right_idxs[part_i]]

    # exited partition (i.e. reached ref point's hypersphere boundary)
    if right_idxs[part_i] >= num_idists or node[0] > idist_max: 
        right_idxs[part_i] = None
        
    if dists_visited and _DEBUG_:
        print('    partition {} outward {} -> {} {}'.format(part_i, search_start_idx, right_idxs[part_i], dists_visited))
    

def _add_neighbor(knn_heap, K_, node, sqdist_node):
    """Maintain a heap of the K_ closest neighbors
    """
    # heapq maintains a "min heap" so we store by -dist
    heap_node = (-sqdist_node, node[1], node[2])
    #print('_add_neighbor: {}'.format(heap_node))
    
    # @TODO: only add neighbor if it isn't in the same cross validation bucket as the query point
    if len(knn_heap) < K_:
        heapq.heappush(knn_heap, heap_node) # @TODO: does this perform an assignment?  does it matter?
        return True
    
    # -knn_heap[0][0] is the (squared) distance to the farthest point in the current knn
    elif sqdist_node < -knn_heap[0][0]:
        heapq.heappushpop(knn_heap, heap_node)
        return True
    
    return False


def _idistance_index(dat, ref_pts, C_):
    """Compute nearest reference point to each point along with farthest point from
    each reference point.  Returns the "iDistance" for each point with reference point
    r_i, i.e. iDistance = i * C * d_ij, where C is the constant C and d is the distance
    from r_i to point j.  Also returns the distance to each reference point's farthest
    point along with its index.
    """
    sorted_idists = []
    partition_dist_max = [0.0] * len(ref_pts)
    partition_point_counts = [0] * len(ref_pts)
    
    for i, mat in enumerate(dat): # for each data file matrix

        for j in xrange(mat.shape[0]): # for each row/point
            
            row = mat[j,:] # @TODO: bind this (and the following) method before the loops
            
            sq = (ref_pts - row)**2 # difference (to each ref point) per dim squared
            
            ssq = np.sum(sq, axis=1) # SSD to each ref point
            
            minr = np.argmin(ssq) # index of closest ref point
            ref_dist = np.sqrt(ssq[minr])
            
            # @TODO: is roundOff needed like it is in the C++ code?
            idist = minr * C_ + ref_dist
            sorted_idists.append((idist, i, j))
            
            partition_point_counts[minr] += 1
            if ref_dist > partition_dist_max[minr]:
#                 if ref_dist > 4.5:
#                     print('ref_dist = {}'.format(ref_dist))
#                     print('ref_pt = {}'.format(ref_pts[minr]))
#                     print('row_pt = {}\n'.format(row))
                partition_dist_max[minr] = ref_dist
        
    # convert idists into a sortable array so that we can perform binary search for
    # a query point with its iDistance rather than searching a B+ tree (given 4e6
    # points the former will be quicker, log2(4e6)=22 comparisons, compared to the
    # latter, 2*50=100 comparisons given 50 dimensions)
    sorted_idists.sort()

    return sorted_idists, partition_dist_max, partition_point_counts


def _reference_points(mins, maxs, dat):
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
    HALF_POINTS = 0
    RANDOM = 1
    
    REF_PT_CHOOSER = HALF_POINTS
    
    D_ = len(mins)
    ref_pts = []
        
    if REF_PT_CHOOSER == HALF_POINTS:
        num_partitions = D_ * 2 # may want to add a center point also
        mids = (mins + maxs) / 2
        for i in xrange(num_partitions):
            ref_pts.append(np.copy(mids)) # need to ensure not using references to the same 'mids' ndarray
        
        for i in xrange(D_):
            ref_pts[i * 2    ][i] = mins[i]
            ref_pts[i * 2 + 1][i] = maxs[i]
            
    elif REF_PT_CHOOSER == RANDOM:
        num_partitions = D_ * 8
        for i in xrange(num_partitions):
            ref_pts.append(dat[-1][-i, :])
        
    else:
        raise ValueError('Unknown REF_PT_CHOOSER')
        
    return ref_pts # [ : len(ref_pts) / 2]


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
    print('Min required C = {}'.format(c))
    
    logc = np.log10(c)
    c = 10**np.ceil(logc)
    return c, maxs, mins