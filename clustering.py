from sklearn.cluster import *
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
    
    
def agglomerative(X, true_clusters, true_labels, title=None):
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=true_clusters)
    clustering.fit(X)
    labels = clustering.labels_
    
    fig_title = 'Ward_Agglomerative_' + title
    plotCluster(X, labels, fig_title)
    return evaluate(labels, true_labels)
    
    
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

    
