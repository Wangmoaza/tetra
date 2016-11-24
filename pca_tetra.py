from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from clustering import *
from divide_db_level import divide

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

def plot2d(result, y, names, title, figName):
    plt.figure()

    for c, i, name in zip("bgrcmykw", list(range(0, len(names))), names):
        plt.scatter(result[y==i, 0], result[y==i, 1], c=c, label=name)
    plt.legend(loc=0, fontsize=10)
    plt.title(title)
    plt.savefig(figName + '.png')
### END - plot2d


def mse(true, pred):
    return np.mean(np.mean(np.square(true - pred), axis=-1))


def perform(rank_num, group):
    # perform PCA for group in ranks[rank_num]
    # e.g. perform PCA for Proteobacteria in rank phylum
    ranks = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    
    X, y, names = load_data(tetraFile='db2_tetra_{0}_{1}.npy'.format(ranks[rank_num], group), 
                            taxonFile='db2_taxon_{0}_{1}.npy'.format(ranks[rank_num], group), 
                            rank=rank_num+1, minCnt=30)

    # PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    print X_2d

    # plot to 2d space
    title = "PCA of TNA of " + group
    figName = 'PCA_' + group
    plot2d(X_2d, y, names, title, figName)
    
    # remap to original space
    X_pred = pca.inverse_transform(X_2d) 
    
    # clustering
    print '...clustering'
    true_clusters = np.unique(y).shape[0] # for cases where there are no 'others'
    d_homo, d_comp, d_vmes, d_ari = dbscan(X_2d, true_clusters, y, title = group + '_pca')
    a_homo, a_comp, a_vmes, a_ari = agglomerative(X_2d, true_clusters, y, title = group + '_pca')
    
    # record
    with open('result_score.txt', 'a') as f:
        general_str = "{rank}\t{group}\t{method}\t{size}\t{mse}\t".format(rank=ranks[rank_num], group=group, method='pca', 
                                                                          size=X_2d.shape[0], mse=mse(X, X_pred))
        d_values_str = "{homo:0.3f}\t{comp:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\t".format(homo=d_homo, comp=d_comp, vmes=d_vmes, ari=d_ari)
        a_values_str = "{homo:0.3f}\t{comp:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\n".format(homo=a_homo, comp=a_comp, vmes=a_vmes, ari=a_ari)
        line = general_str + d_values_str + a_values_str
        f.write(line)
    ### END - with open
### END - perform


def main():
    ranks = ["domain", "phylum", "class", "order", "family", "genus", "species"]

    # loop from phylum to family
    for i in range(1, len(ranks)-2):

        # generate taxon and tetra file for rank under this rank
        # e.g. if current rank is class, then generate datasets for order
        names = np.load('db2_{0}_names.npy'.format(ranks[i])) # e.g. ranks[i] == class
        under_rank_list = []
        for group in names:
            taxonFile = 'db2_taxon_{0}_{1}.npy'.format(ranks[i], group)
            tetraFile = 'db2_tetra_{0}_{1}.npy'.format(ranks[i], group)
            if i < 3:
                under_ranks = divide(i+1, taxonFile, tetraFile, 300) # e.g. divide by order
            else:
                under_ranks = divide(i+1, taxonFile, tetraFile, 100)
            under_rank_list = under_rank_list + under_ranks
        ### END - for group
        
        labelName = 'db2_{0}_names'.format(ranks[i+1]) # e.g. save order names
        np.save(labelName, under_rank_list)

        # peform pca
        for group in names:
            print '\n'
            print '******** ' + group + ' ********'
            perform(i, group)
        ### END - for group
        
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
