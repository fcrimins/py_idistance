#!/usr/bin/python
from __future__ import division
import sys
import os
import numpy as np
import idist


def main(argv):
    
    if len(argv) < 2 or len(argv) > 4:
        raise ValueError('argv must be of length between 2 and 4 (argv = {}'.format(argv))

    model_dir = argv[1]
    print('model_dir = {}'.format(model_dir))
    ls = sorted(os.listdir(model_dir))
    print('ls = {}'.format(ls))
    
    dat = []
    for f in ls:
        p = os.path.join(model_dir, f)
        dat.append(np.load(p))
        #print('{} =\n{}\n'.format(p, dat[-1]))
        
    bplus_tree = idist.bplus_tree(dat, float(argv[2]) if len(argv) > 2 else None, float(argv[3]) if len(argv) > 3 else None)
        
        
    
    
    

if __name__ == "__main__":
    main(sys.argv)