from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from clustering import *

# load dataset
def load_data(tetraFile, taxonFile, rank, minCnt=10):
    X = np.load(tetraFile)
    taxons = np.load(taxonFile)
    names = getTopNames(taxons, rank, minCnt)
    y = label2num(taxons, names, rank) 
    return X, y, names
### END - def load_data


def getTopNames(taxons, rank, minCnt):
    # make dictionary {phylumName: count}
    dic = {}
    for item in taxons[:, rank]:
        dic[item] = dic.get(item, 0) + 1

    names = []
    for label in dic.keys():
        if dic.get(label) >= minCnt:
            names.append(label)
            print (label, dic.get(label))
    ### END - for label

    names.append('Others')

    return names
### END - getTopNames


def label2num(taxons, names, rank):
    # make list of phylum names in index 0: Bacteriodetes, etc.
    labels = taxons[:, rank]
    indexList = []
    
    for i in range(len(labels)):
        item = labels[i]
        if item in names:
            indexList.append(names.index(item))
        else:
            indexList.append(len(names)-1) # last element is 'Others'
    ### END - for i
        
    for i in range(len(names)):
        print i, indexList.count(i)
    
    return np.asarray(indexList)
### END - def label2num

def plot(result, y, names, title, figName):
    plt.figure()

    for c, i, name in zip("bgrcmykw", list(range(0, len(names))), names):
        plt.scatter(result[y==i, 0], result[y==i, 1], c=c, label=name)
    plt.legend(loc=0, fontsize=10)
    plt.title(title)
    plt.savefig(figName + '.png')
### END - plot


def mse(true, pred):
    return np.mean(np.mean(np.square(true - pred), axis=-1))


def main():
    phylum_names = np.load('db2_phylum_names.npy')
    
    for phy in phylum_names[:-1]:
        print '******** ' + phy + ' ********'
        X, y, names = load_data(tetraFile='db2_tetra_phylum_' + phy +'.npy', 
                                taxonFile='db2_taxon_phylum_' + phy + '.npy', 
                                rank=2, minCnt=30)

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        print X_2d

        # plot to 2d space
        title = "PCA of TNA of " + phy
        figName = 'PCA_' + phy
        plot(X_2d, y, names, title, figName)
        
        X_pred = pca.inverse_transform(X_2d) # remap to original space
        
        # clustering
        true_clusters = np.unique(y).shape[0] # for cases where there are no 'others'
        d_homo, d_comp, d_vmes, d_ari = dbscan(X_2d, true_clusters, y, title = phy + '_pca')
        a_homo, a_comp, a_vmes, a_ari = agglomerative(X_2d, true_clusters, y, title = phy + '_pca')
        
        # record
        with open('result_score.txt', 'a') as f:
            general_str = "{group}\t{method}\t{size}\t{mse}\t".format(group=phy, method='pca', size=X_2d.shape[0], mse=str(mse(X, X_pred)))
            d_values_str = "{homo:0.3f}\t{comp:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\t".format(homo=d_homo, comp=d_comp, vmes=d_vmes, ari=d_ari)
            a_values_str = "{homo:0.3f}\t{comp:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\n".format(homo=a_homo, comp=a_comp, vmes=a_vmes, ari=a_ari)
            line = general_str + d_values_str + a_values_str
            f.write(line)
        
        # perform clustering
        
        # TODO
    """
    X, y, names = load_data(tetraFile='db2_tetra_top.npy', 
                                taxonFile='db2_taxons_top.npy', 
                                rank=1, minCnt=100)

    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    reconstruct = pca.inverse_transform(result)
    print reconstruct.shape
    print "MSE: " + str(mse(X, reconstruct))
    """
    
    
main()
