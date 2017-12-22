from Bio.Phylo.TreeConstruction import _DistanceMatrix, DistanceTreeConstructor
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import Bio.Phylo as Phylo

def construct_tree(X_2d, acc, title):
    acc = list(acc)
    data = pairwise_distances(X_2d).astype('float')
    data[np.isnan(data)] = 0
    data_list = []
    for i in range(data.shape[0]):
        #for j in range(i, data.shape[0]):
        data_list.append([data[i, j] for j in range(0, i+1)])
    data = data_list
    dm = _DistanceMatrix(acc, matrix=data)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)
    Phylo.write(tree, title + ".nwk", 'newick')

def main():
    methods = ["AE", "AE-tSNE", "PCA", "PCA-tSNE", "MDS", "SOM"]
    acc = np.load("selected21_treelabel.npy")
    for method in methods:
        X_2d = np.load("selected21_{0}.npy".format(method))
        title = "selected21_{0}_nj".format(method)
        construct_tree(X_2d, acc, title)

main()


