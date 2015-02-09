import numpy as np


def bplus_tree(dat):
    
    c_constant, maxs, mins = _calculate_c(dat)
    
    ref_pts = _reference_points(maxs, mins)
    
    idists, partition_dist_max, partition_dist_max_idx = _idistance_index(dat, ref_pts, c_constant)
    
    
    return 0


def _idistance_index(dat, ref_pts, c_constant):
    """Compute nearest reference point to each point along with farthest point from
    each reference point.  Returns the "iDistance" for each point with reference point
    r_i, i.e. iDistance = i * C * d_ij, where C is the constant C and d is the distance
    from r_i to point j.  Also returns the distance to each reference point's farthest
    point along with its index.
    """
    idists = []
    partition_dist_max = [0.0] * len(ref_pts)
    partition_dist_max_idx = [None] * len(ref_pts)
    
    for i, mat in enumerate(dat): # for each data file matrix
        idists.append([])

        for j in xrange(mat.shape[0]): # for each row/point
            
            row = mat[j,:] # TODO: bind this (and the following) method before the loops
            
            sq = (ref_pts - row)**2 # difference (to each ref point) per dim squared
            
            ssq = np.sum(sq, axis=1) # SSD to each ref point
            
            minr = np.argmin(ssq) # index of closest ref point
            ref_dist = np.sqrt(ssq[minr])
            
            # TODO: is roundOff needed like it is in the C++ code?
            idists[-1].append(minr * c_constant + ref_dist)
            
            if ref_dist > partition_dist_max[minr]:
                partition_dist_max[minr] = ref_dist
                partition_dist_max_idx[minr] = (i, j)

    return idists, partition_dist_max, partition_dist_max_idx


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