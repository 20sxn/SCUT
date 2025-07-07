import sys, os, argparse
import numpy as np

import pickle
from sklearn.utils.extmath import randomized_svd, make_nonnegative
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from ete3 import Tree

class Counter:
    def __init__(self,init_val=-1):
        self.count = init_val

    def get_new_id(self):
        self.count += 1
        return self.count

def SpectralBiCoClustering(data,random_state=None,n_components=2):
    """
    extract both fiedler vector from "data"
    """
    data[data <= 0] = 1e-10
    row_diag = np.asarray(1.0 / np.sqrt(data.sum(axis=1))).squeeze()
    col_diag = np.asarray(1.0 / np.sqrt(data.sum(axis=0))).squeeze()
    row_diag = np.where(np.isnan(row_diag), 0, row_diag)
    col_diag = np.where(np.isnan(col_diag), 0, col_diag)
    
    an = row_diag[:, np.newaxis] * data * col_diag
    an = np.where(np.isnan(an), 1, an)
    an = np.where(np.isinf(an), 1, an)
    
    U, S, Vt = randomized_svd(an, 
                              n_components=n_components,
                              n_iter=10,
                              random_state=random_state)

    Fiedler_U = U[:,1]
    Fiedler_V = Vt[1]
    
    return Fiedler_U, Fiedler_V,S[1],col_diag

def th_selection(th_mode,lendata,n_min,K,U,n_s,mode):
    """
    Select the threshold
    """
    th = None
    if th_mode == "kde" and lendata >= n_min:
        const = int(K/ (U[:, np.newaxis].max() - U[:, np.newaxis].min()))
        X = U[:, np.newaxis]*const #scale of data seems to be important for kde (bandwidth)
        X_eval = np.linspace(X.min(), X.max(), n_s)[:, np.newaxis] 
        kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(X)
        log_dens = kde.score_samples(X_eval)
        
        try:
            if mode == "min": #smallest local min of density
                argm = argrelextrema(log_dens,np.less)[0]

                th_id = argm[log_dens[argm].argmin()]
                th = (X_eval/const)[th_id][0]

            if mode == "valley": #search for 2 high density peaks and the deepest valley between them
                loc_min = argrelextrema(log_dens,np.less)[0]
                loc_max = argrelextrema(log_dens,np.greater)[0]
                
                local_extrema = [loc_max[0]]
                for i in range(len(loc_min)):
                    local_extrema.append(loc_min[i])
                    local_extrema.append(loc_max[i+1])
                
                left_dist = [] #highest peak at the left of the valley
                curr_l_max = log_dens[local_extrema][0]
                for i in log_dens[local_extrema]:
                    if i > curr_l_max:
                        curr_l_max = i
                    left_dist.append(curr_l_max-i)
                
                right_dist = [] #highest peak at the right of the valley
                curr_r_max = log_dens[local_extrema][-1]
                for i in log_dens[local_extrema][::-1]:
                    if i > curr_r_max:
                        curr_r_max = i
                    right_dist.append(curr_r_max-i)
                
                dist_prod = np.array(left_dist)*np.array(right_dist[::-1])
                
                th_id = loc_min[(dist_prod.argmax()-1)//2]
                th = (X_eval/const)[th_id][0]
        except:
            pass
        
    if th == None: #search for largest gap between datapoints in the projection
        u_cpy = U.copy()
        u_cpy.sort()
        idx = np.argmax(u_cpy[1:]-u_cpy[:-1])
        th = (u_cpy[idx+1]+u_cpy[idx])/2
    return th

#TODO Do not pass down the full "data" matrix each time, only a pointer to the original one and idx to access the subset
def _RecClusterDataToTree(data,seq_ids,treenode,counter,distsum,th_mode="simple",K=10,mode='valley',n_min=64,n_s=1000,seed=None):
    """
    Recursively build a clustering dendrogram
    """
    U,V,S,cd = SpectralBiCoClustering(data,random_state=seed)

    lendata = len(data) 
    th = th_selection(th_mode,lendata,n_min,K,U,n_s,mode)
    
    neg_idx = np.where(U < th)[0]
    pos_idx = np.where(U >= th)[0]
    
    assert (len(neg_idx) > 0)
    assert (len(pos_idx) > 0)

    neg_seq_ids = [seq_id for e,seq_id in enumerate(seq_ids) if (e in neg_idx)]
    pos_seq_ids = [seq_id for e,seq_id in enumerate(seq_ids) if (e in pos_idx)]

    treenode.add_feature("V",V)
    treenode.add_feature("U",U)
    treenode.add_feature("th",th)
    treenode.add_feature("S",S)
    treenode.add_feature("cd",cd)

    if (len(neg_idx) > 1): 
        name = "I"+str(counter.get_new_id())
        neg_node = treenode.add_child(name=name)
        neg_distsum = (1-data[neg_idx]).sum()
        neg_node.dist = distsum - neg_distsum 
        _RecClusterDataToTree(data[neg_idx],neg_seq_ids,neg_node,counter,neg_distsum,
                              th_mode=th_mode,K=K,mode=mode,n_min=n_min,n_s=n_s,seed=seed)
        
    if (len(neg_idx) == 1):
        neg_node = treenode.add_child(name=neg_seq_ids[0])
        neg_distsum = (1-data[neg_idx]).sum()
        neg_node.dist = distsum

    
    if (len(pos_idx) > 1):
        name = "I"+str(counter.get_new_id())
        pos_node = treenode.add_child(name=name)
        pos_distsum = (1-data[pos_idx]).sum()
        pos_node.dist = distsum - pos_distsum
        _RecClusterDataToTree(data[pos_idx],pos_seq_ids,pos_node,counter,pos_distsum,
                              th_mode=th_mode,K=K,mode=mode,n_min=n_min,n_s=n_s,seed=seed)

    if (len(pos_idx) == 1):
        pos_node = treenode.add_child(name=pos_seq_ids[0])
        pos_distsum = (1-data[pos_idx]).sum()
        pos_node.dist = distsum


def ClusterDataToTree(data,seq_ids,fname = "tree.treefile",pickle_name = "",th_mode="simple",mode="valley",K=10,n_min=64,n_s=1000,seed=None):
    """
    Build clustering dendrogram and save it in extended newick format and pickle format
    """
    counter = Counter()
    root = Tree()
    n = len(data)
    distsum = (1-data).sum()
    _RecClusterDataToTree(data,seq_ids,root,counter,distsum,th_mode=th_mode,K=K,mode=mode,n_min=n_min,n_s=n_s,seed=seed)
    #we divide the distance by n so that it scales with n and not n^2 (easier for iTOL)
    for node in root.traverse():
        node.dist = node.dist/n

    if fname != "":
        root.write(format=1,outfile=fname)
    if pickle_name != "":
        with open(pickle_name, 'wb') as handle:
            pickle.dump(root, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return root


def infer_leaf_from_tree_TH(tree,rep):
    """
    infer corresponding leaf in tree of the representation given as input
    tree should be the output of ClusterDataToTree()
    """
    curr_node = tree

    while not curr_node.is_leaf():
        left,right = curr_node.children
        V = curr_node.V
        th = curr_node.th
        S = curr_node.S
        cd = curr_node.cd
        
        rd = np.asarray(1.0 / np.sqrt(rep.sum(axis=-1))).squeeze()
        rd = np.where(np.isnan(rd), 0, rd)

        rep_n = rd * rep * cd
        
        val = (V * rep_n/S).sum()-th
        
        if val < 0:
            curr_node = left
        else:
            curr_node = right
    return curr_node

