from sklearn.cluster import *
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.colors as colors

import numpy as np

def dbscan(X, true_clusters, true_labels, title=None):
    db = DBSCAN(eps=1, min_samples=30)
    db.fit(X)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    
    fig_title = 'DBSCAN_' + title
    plotCluster(X, labels, fig_title)
    return evaluate(labels, true_labels)
    
    
def agglomerative(X, true_clusters, true_labels, title=None, connect=True, linkage='average'):
    if connect:
        knn_graph = kneighbors_graph(X, 30, include_self=False)
        model = AgglomerativeClustering(linkage=linkage, n_clusters=true_clusters, connectivity=knn_graph)
    
    else:
        model = AgglomerativeClustering(linkage=linkage, n_clusters=true_clusters)
    
    model.fit(X)
    labels = model.labels_
    fig_title = 'Agglomerative_' + title
    plotCluster(X, labels, fig_title)

    evalList = []
    evalList.append(evaluate(labels, true_labels))
                    
    """
    plt.figure(figsize=(10,8))
    for connectivity in (None, knn_graph):
        for index, linkage in enumerate(linkages):
            plt.subplot(2, 3, index + 1)
            model = AgglomerativeClustering(linkage=linkage, n_clusters=true_clusters)
            model.fit(X)
            labels = model.labels_
            print "...", linkage
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.spectral)
            plt.title('linkage={0}'.format(linkage), fontdict=dict(verticalalignment='top'))
            plt.axis('equal')
            plt.axis('off')
            plt.subplots_adjust(bottom=0, top=.89, wspace=0, left=0, right=1)
            plt.suptitle('%s, connectivity=%r' %
                        (title, connectivity is not None), size=17)
            evalList.append(evaluate(labels, true_labels))
        ### END - for index, linkage
    ### END - for connectivity
    
    fig_title = 'Agglomerative_' + title
    plt.savefig(fig_title)
    """
    
    return evalList
    

def evaluate(labels, true_labels):
    homo = metrics.homogeneity_score(true_labels, labels)
    comp = metrics.completeness_score(true_labels, labels)
    vmes = metrics.v_measure_score(true_labels, labels)
    ari = metrics.adjusted_rand_score(true_labels, labels)
    return homo, comp, vmes, ari
        
    
def plotCluster(X, y, title):
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 10)
    
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], color=colors[y].tolist())
    plt.title(title)
    plt.savefig(title + '.png')

    
