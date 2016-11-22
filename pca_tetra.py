from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

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
            indexList.append(len(names)-1) # last element is names is 'Others'
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
    plt.savefig(figName)
### END - plot

def mse(true, pred):
    return np.mean(np.mean(np.square(true - pred), axis=-1))

def main():
    phylum_names = np.load('db2_phylum_names.npy')
    """
    for phy in phylum_names[:-1]:
        print '******** ' + phy + ' ********'
        X, y, names = load_data(tetraFile='db2_tetra_phylum_' + phy +'.npy', 
                                taxonFile='db2_taxon_phylum_' + phy + '.npy', 
                                rank=2, minCnt=30)

        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        print result

        title = "PCA of TNA of " + phy
        figName = 'PCA_' + phy + '.png'
        plot(result, y, names, title, figName)

        reconstruct = pca.inverse_transform(result)
        print reconstruct.shape
        print "MSE: " + str(mse(X, reconstruct))
        print
    """
    X, y, names = load_data(tetraFile='db2_tetra_top.npy', 
                                taxonFile='db2_taxons_top.npy', 
                                rank=1, minCnt=100)

    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    reconstruct = pca.inverse_transform(result)
    print reconstruct.shape
    print "MSE: " + str(mse(X, reconstruct))
    
main()
