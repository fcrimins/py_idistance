import numpy as np


def bplus_tree(dat):
    
    c_constant, maxs, mins = _calculate_c(dat)
    
    ref_pts = _reference_points(maxs, mins)
    
    idists = _idistance_index(dat, ref_pts, c_constant)
    
    
    return 0


def _idistance_index(dat, ref_pts, c_constant):

    idists = []
    partition_dist_max = [0.0] * len(ref_pts)
    partition_dist_max_index = [None] * len(ref_pts)
    
    for i, mat in enumerate(dat): # for each data file matrix
        idists.append([])

        for j in xrange(mat.shape[0]): # for each row/point
            
            row = mat[j,:] # TODO: bind this method before the loops
            
            subtr = (ref_pts - row)**2 # difference per dim squared
            
            sq_dists = np.sum(subtr, axis=1)
            
            min_ref_idx = np.argmin(sq_dists)
            
            # TODO: is roundOff needed like it is in the C++ code?
            idists[-1].append(min_ref_idx * c_constant + np.sqrt(sq_dists[min_ref_idx]))
            
            
            
            
#             
#             np.argmin(
# 
#             for k, rp in enumerate(ref_pts): # for each reference point
# 
#                 d = np.sum(mat[j,:] - rp)**2)
#             
#         
#         ref_dist = dist(&datapoints[offset],&reference_points[0]);
#         
#         //if(idp) { cout << "checking point " << i << endl; }
#         
#         //if(idp) { cout << "p 0: " << ref_dist << endl; }
#          
#         //check each partition (baseline from partition 0)
#         for(int part=1; part < number_partitions; part++)
#         {
#             //get distance to this partition
#             temp_dist = dist(&datapoints[offset],&reference_points[part*number_dimensions]);
#         
#             //if(idp) { cout << "p " << part << ": " << temp_dist << endl; }
#             
#             if(temp_dist < ref_dist) //if closer to this partition
#             {
#                 //if(idp) { cout << "  - updating from " << current_partition << endl; }
#                 
#                 current_partition = part; //update
#                 ref_dist = temp_dist;
#                 
#             }
#         }
#         
#         
#         //if(idp) { cout << "final part = " << current_partition << " : " << ref_dist << endl; }
#         
#         //calculate key index value
#         //datapoint_index[i] = current_partition*constant_c + ref_dist;
#         
#         //updated to try to fix double precision issues
#         temp_dist = roundOff(current_partition*constant_c + ref_dist);
#         
#         //if(idp) { cout << "  index_dist = " << temp_dist << endl; }
#         
#         datapoint_index[i] = temp_dist;
#         
#         //cout << "  DATA INDEX: " << datapoint_index[i] << " P: " <<
#         //  current_partition << " D: " << ref_dist << endl;
#         
#         //also need to update partition dist_max
#         if(ref_dist > partition_dist_max[current_partition])
#         {
#             partition_dist_max[current_partition] = ref_dist;
#             partition_dist_max_index[current_partition] = i;
#         }
#     }
# }





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