import numpy as np


def bplus_tree(dat):
    
    c = _calculate_c(dat)
    
    
    return 0



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
    return c