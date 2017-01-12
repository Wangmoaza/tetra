import numpy as np
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# load dataset
def load_data(tetraFile, taxonFile, rank, minCnt=10):
    X = np.load(tetraFile)
    
    # standardize data
    X_scaled = preprocessing.scale(X) + 1
    
    taxons = np.load(taxonFile)
    names = getTopNames(taxons, rank, minCnt)
    y = label2num(taxons, names, rank) 
    
    # exclude Others
    condition = y != len(names) - 1
    X_nonOther = np.ndarray(shape=(0, X.shape[1]), dtype=int)
    for i in range(X.shape[0]):
        if condition[i]:
            X_nonOther = np.vstack((X_nonOther, X_scaled[i, ]))
            
    y_nonOther = np.extract(condition, y)
    print X_nonOther.shape, y_nonOther.shape
    return X_nonOther, y_nonOther, names[:-1]
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