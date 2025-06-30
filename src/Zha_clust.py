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



def _RecClusterDataToTreeZha(data,seq_ids,treenode,counter,distsum):
    """
    Recursively build a clustering dendrogram
    """
    U,V,S,cd = SpectralBiCoClustering(data)

    th = 0
    
    neg_idx = np.where(U < th)[0]
    pos_idx = np.where(U >= th)[0]
    
    neg_feat_idx = np.where(V < th)[0]
    pos_feat_idx = np.where(V >= th)[0]
    
    assert (len(neg_idx) > 0)
    assert (len(pos_idx) > 0)

    neg_seq_ids = [seq_id for e,seq_id in enumerate(seq_ids) if (e in neg_idx)]
    pos_seq_ids = [seq_id for e,seq_id in enumerate(seq_ids) if (e in pos_idx)]

    treenode.add_feature("V",V)
    treenode.add_feature("U",U)
    treenode.add_feature("th",th)
    treenode.add_feature("S",S)
    treenode.add_feature("cd",cd)
    
    dist_cut = (1-data[neg_idx][:,pos_feat_idx]).sum() + (1-data[pos_idx][:,neg_feat_idx]).sum()
    
    neg_distsum = (1-data[neg_idx][:,neg_feat_idx]).sum() + dist_cut/2
    pos_distsum = (1-data[pos_idx][:,pos_feat_idx]).sum() + dist_cut/2
    

    
    if (len(neg_idx) > 1) and (len(neg_feat_idx) > 1):
        name = "I"+str(counter.get_new_id())
        neg_node = treenode.add_child(name=name)
        #neg_distsum = (1-data[neg_idx][:,neg_feat_idx]).sum() + dist_cut/2
        neg_node.dist = distsum - neg_distsum #- dist_cut/2
        _RecClusterDataToTreeZha(data[neg_idx][:,neg_feat_idx],neg_seq_ids,neg_node,counter,neg_distsum)

    if (len(pos_idx) > 1) and (len(pos_feat_idx) > 1):
        name = "I"+str(counter.get_new_id())
        pos_node = treenode.add_child(name=name)
        #pos_distsum = (1-data[pos_idx][:,pos_feat_idx]).sum() + dist_cut/2
        pos_node.dist = distsum - pos_distsum #- dist_cut/2
        _RecClusterDataToTreeZha(data[pos_idx][:,pos_feat_idx],pos_seq_ids,pos_node,counter,pos_distsum)
    
    if (len(neg_feat_idx) == 1) and (len(neg_idx) > 1):
        name = "I"+str(counter.get_new_id())
        neg_node = treenode.add_child(name=name)
        #neg_distsum = (1-data[neg_idx]).sum() + dist_cut/2
        neg_node.dist = distsum - neg_distsum 
        for i in range(len(neg_seq_ids)):
            neg_leaf = neg_node.add_child(name=neg_seq_ids[i])
            #neg_distsum = (1-data[neg_idx[i]]).sum()
            neg_leaf.dist = neg_distsum

            
    if (len(pos_feat_idx) == 1) and (len(pos_idx) > 1):
        name = "I"+str(counter.get_new_id())
        pos_node = treenode.add_child(name=name)
        #pos_distsum = (1-data[pos_idx]).sum() + dist_cut/2
        pos_node.dist = distsum - pos_distsum 
        for i in range(len(pos_seq_ids)):
            pos_leaf = pos_node.add_child(name=pos_seq_ids[i])
            #pos_distsum = (1-data[pos_idx[i]]).sum()
            pos_leaf.dist = pos_distsum

            
    if (len(neg_feat_idx) >= 1) and (len(neg_idx) == 1):
        neg_node = treenode.add_child(name=neg_seq_ids[0])
        neg_distsum = (1-data[neg_idx]).sum()
        neg_node.dist = distsum

        
    if (len(pos_feat_idx) >= 1) and (len(pos_idx) == 1):
        pos_node = treenode.add_child(name=pos_seq_ids[0])
        pos_distsum = (1-data[pos_idx]).sum() 
        pos_node.dist = distsum 


def ClusterDataToTreeZha(data,seq_ids,fname = "tree.treefile",pickle_name = ""):
    """
    Build clustering dendrogram and save it in extended newick format and pickle format
    """
    counter = Counter()
    root = Tree()
    distsum = (1-data).sum()

    _RecClusterDataToTreeZha(data,seq_ids,root,counter,distsum)
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
        
        if val <= 0:
            curr_node = left
        else:
            curr_node = right
    return curr_node

