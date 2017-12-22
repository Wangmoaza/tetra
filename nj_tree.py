import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from skbio import DistanceMatrix
from skbio.tree import nj

def construct_tree(X_2d, acc, title):
    data = pairwise_distances(X_2d)
    data[np.isnan(data)] = 0
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            data[j, i] = data[i, j]
    
    dm = DistanceMatrix(data, acc)
    tree = nj(dm)
    newick_str = nj(dm, result_constructor=str)
    
    with open(title + ".nwk", "w") as f:
        f.write(newick_str)

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def main():
    methods = ["AE", "AE-tSNE", "PCA", "PCA-tSNE", "MDS", "SOM"]
    acc = np.load("selected21_ids.npy")
    for method in methods:
        X_2d = np.load("selected21_{0}.npy".format(method))
        title = "selected21_{0}_nj".format(method)
        construct_tree(X_2d, acc, title)

main()
