#!/usr/bin/python
from __future__ import division
import sys
import os
import numpy as np
import idist


def main(argv):
    
    if len(argv) != 2 and len(argv) != 3:
        raise ValueError('argv must be of length 2 or 3 (argv = {}'.format(argv))

    model_dir = argv[1]
    print('model_dir = {}\n'.format(model_dir))
    ls = sorted(os.listdir(model_dir))
    print('ls = {}\n'.format(ls))
    
    dat = []
    for f in ls:
        p = os.path.join(model_dir, f)
        dat.append(np.load(p))
        #print('{} =\n{}\n'.format(p, dat[-1]))
        
    bplus_tree = idist.bplus_tree(dat, float(argv[2]) if len(argv) == 3 else None)
        
        
    
    
    

if __name__ == "__main__":
    main(sys.argv)