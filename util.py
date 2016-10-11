import scipy.stats as st
import numpy as np

def spearmanr_nonan(x,y):
    '''
    same as scipy.stats.spearmanr, but if all values are unique, returns 0 instead of nan
    (Output: rho, pval)
    '''
    r, p = st.spearmanr(x, y)
    if np.isnan(p):
        if len(np.unique(x))==1 or len(np.unique(y))==1:
            print "WARNING: spearmanr is nan due to unique values, setting to 0"
            p = 0.0
            r = 0.0
        else:
            raise Exception("found nan spearman")
    assert not np.isnan(r)
    return r, p