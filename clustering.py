from sklearn.cluster import *
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

def dbscan(true_clusters, true_labels, data):
    db = DBSCAN(eps=0.3, min_samples=30).fit(X)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    # TODO: record and plot

def agglomerative(true_clusters, true_labels, data):
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=true_clusters)
    clustering.fit(data)
    


def record(labels, true_labels):
    # TODO
    # write to file
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))

def plotCluster():
    # TODO
    # implement