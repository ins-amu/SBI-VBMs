import numpy as np
from scipy import stats
from scipy.stats import kurtosis


def trip(edges_arr,n):
    mat=np.zeros((n,n))
    mat[np.triu_indices(n,1)]=edges_arr
    mat=mat+mat.T+np.eye(n)
    return mat

def go_edge(tseries):
    '''  
    tseries: [nT, nROI]
    '''
    nregions=tseries.shape[1]
    Blen=tseries.shape[0]
    nedges=int(nregions**2/2-nregions/2)
    iTriup= np.triu_indices(nregions,k=1) 
    gz=stats.zscore(tseries)
    Eseries = gz[:,iTriup[0]]*gz[:,iTriup[1]]
    return Eseries

def calculate_FC(bold):
    ''' 
    bold: [nT, nROI]
    '''
    mregions = bold.shape[1]
    edge_ts = go_edge(bold)
    fc_tri = np.mean(edge_ts, axis=0)
    fc = trip(fc_tri, mregions)
    return fc

def calculate_dFC(bold):
    '''
    bold: [nT, nROI]
    '''
    edge_ts = go_edge(bold)
    dfc = np.corrcoef(edge_ts)
    return dfc

def local_to_global_coherence(edg_ts, mregions, roi_indices=[]):
    tlen, medges = edg_ts.shape
    
    Mask=np.arange(medges)
    Mask=(trip(Mask,mregions)-2*np.identity(mregions)).astype(int)
    
    # C=np.zeros((mregions,tlen))
    C = []
    if len(roi_indices) == 0:
        roi_indices = range(mregions)
    for i in roi_indices:
        edg_reg=Mask[i]
        edg_reg=edg_reg[edg_reg!=-1] # list of edges attached to node (labels)
        egd_ts_reg=edg_ts[:,edg_reg]
        C.append(np.mean(egd_ts_reg,axis=1))
    C = np.array(C)
    if len(roi_indices) == 1:
        C = C.reshape(1, -1)

    return C.mean(axis=1), C.std(axis=1), kurtosis(C, axis=1)