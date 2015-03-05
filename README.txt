
README.txt
This file contains some notes that were taken while developing py_idistance.


Install:
$ python setup.py install


Run:
$ python pydist.py <directory_containing_numpy_matrices> [<iDistance_initial_radius>] [<K>]
e.g.
$ python pydist.py ../data/7 0.05 5


To create a numpy data file:
>>> import numpy as np
>>> X = np.random.randn((1e6, 50))
>>> np.save('../data/7/0', X) # creates 0.npy file in ../data/7 directory

_______________________________________________
2/25/15

todo

Q: figure out why iDistance is visiting so many nodes when D is 50
A: too much overlapping!!!!

Q: why is it all of a sudden inside so many partitions when radius gets to 0.4, but not at 0.2
A: 0.2 is never investigated

Q: why is radius getting all the way up to 1.8 when it looks like most points are within 4 of their
	partitions?
	
13 partitions, out of 24 have the point inside
is this because we're selecting partition points way outside the cloud?  which makes for larger
	distances to farthest points for each partition?
	
there should be a way to measure the amount of overlap between partitions, which there is
	for each point, count the number of partitions its in OR count average partition radius
	
why can't it trim any partitions?

todo: next count avg number of partitions visited in an iDistance query
_______________________________________________
2/26/15

idea: kd-tree doesn't work in high dimensions b/c 1 dimension doesn't say much about distance between
points.  but that's because query points often fall on the boundaries between splits.  what if instead the
space was partitioned into thirds at each level of the tree where only 1 of the 3 was trimmed.  this
trimmed piece would be much less likely to have any of the NN because the query point wouldn't
ever occur nearer than 1/6 of the span of the dimension to a split.

e.g. P1: 0 1 2 3 <SPLIT> P2: 4 5 6 7 <SPLIT> P3: 8 9 10 11
Q 2 -> trim P3 -> dist to split 7.5 - 2 = 5.5 > 2 (= 12 / 6)
Q 6 -> trim P1 -> dist to split 6 - 3.5 = 2.5 > 2

if K were big enough however, this would still have problems, but that would be the case for any
algorithm with big enough K.

in practice maybe we don't want to split into thirds but rather we want to minimize the chances
of having to re-visit a trimmed 1/m'th of the space.  this will depend on K but also D (and N).
D tells us how much we know from looking at a single dimension.  solve:

	P(x is one of q's KNN | abs(x_i - q_i) < t)
_______________________________________________
2/26/15

idea: could always just do PCA and then select closest KNN ignoring the tiny PCs.  this would be like hashing using the larger PCs, though not "locality sensitive".
 
Locality Sensitive Hashing
http://stackoverflow.com/questions/5751114/nearest-neighbors-in-high-dimensional-data
"all current indexing techniques (based on space partitioning) degrade to linear search for sufficiently high dimensions [1][2][3]"

looks like it's available in sklearn version 0.16
http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.LSHForest.html

_______________________________________________
As dimensionality increases an n-ball grows much slower (in volume) than a hyper-cube of
the same diameter.  This means that points become much farther apart as the number of 
dimensions increases.  What this means for iDistance is that the k'th farthest distance
approaches the average partition size so all partitions end up needing to be searched and
the algorithm degrades to sequential search (as mentioned in the stackoverflow link above).
_______________________________________________

idea:
1. compute the (approximate) percentile rank along each dimension for a query point
2. build D kd-trees each starting with each dimension (or D^2 where first 2 levels are
	eacy combination of 2 dimensions)
3. find the 1 (or 2) dimension(s) with the most extreme percentile ranks and choose
	that kd-tree to perform the search
	
generally, this idea of beefing up the preprocessing doesn't seem explored

What percentage of a query point's knn are in the right kd tree top level partition given the query point lies at the pth percentile of that dimension? (Note also that dimension scaling may trump query point percentiles.)
_______________________________________________

idea:
From the second answer to the stackoverflow link above (which mentions using Voronoi Tessellations).
Rather than computing the Voronoi, just select 250 reference points at random and assign all of the
other points to their nearest.  Possibly do this in multiple layers.  Then for querying, select the
closest reference point and search its sub-points.  How do you trim then, though?
	