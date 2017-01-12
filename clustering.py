from sklearn.cluster import *
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
import numpy as np


class Clustering:
    def __init__(self, name, method, X, y):
        self.name = name
        self.method = method
        self.X = X
        self.y = y
        self.db = None  # predicted label in tuple
        self.agglo = None  # predicted label in tuple
    ### END - def __init__


    def dbscan(self, min_samples):
        model = DBSCAN(eps=1, min_samples=min_samples)
        model.fit(self.X)
        self.db = (model.labels_,)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(self.db[0])) - (1 if -1 in self.db[0] else 0)
        print('Estimated number of clusters: %d' % n_clusters_)

        return self.evaluate(self.db)
    ### END - def dbscan


    def agglomerative(self, connect=True, linkage='ward'):
        # connectivity constrain
        if connect:
            knn_graph = kneighbors_graph(self.X, 5)
        else:
            knn_graph = None

        if linkage in ('ward', 'average', 'complete'):
            model = AgglomerativeClustering(linkage=linkage,
                                            n_clusters=self.y.shape[0],
                                            connectivity=knn_graph)
            model.fit(self.X)
            self.agglo = (model.labels_,)
        ### END - if linkage

        elif linkage == 'all':
            label_list = []
            for linkage in ('ward', 'average', 'complete'):
                model = AgglomerativeClustering(linkage=linkage,
                                                n_clusters=self.y.shape[0],
                                                connectivity=knn_graph)
                model.fit(self.X)
                label_list.append(model.labels_)
            ### END - for linkage
            self.agglo = tuple(label_list)
        ### END - elif linkage

        else:
            print("Error: Wrong linkage argument")
            return
        ### END - else

        return self.evaluate(self.agglo)
    ### END - def agglomerative


    def plotCluster(self, alg='all'):
        titles = ['True Clusters', 'DBSCAN', 'Ward', 'Average', 'Complete']
        labels = [self.y, self.db[0]]
        for lab in self.agglo:
            labels.append(lab)

        # plot all clustering alg
        if alg == 'all':
            plt.figure()
            for i, title, label in zip(range(len(titles)), titles, labels):
                plt.subplot(2, 3, i + 1)
                plt.scatter(self.X[:, 0], self.X[:, 1], c=label, cmap=plt.cm.spectral)
                plt.title(title, fontdict=dict(verticalalignment='top'))
                plt.axis('equal')
                plt.autoscale()
                plt.axis('off')
            ### END - for i, title, label

            plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
            plt.suptitle('Clustering of {0}'.format(self.name), size=17)
            plt.savefig('Clustering_{0}_{1}.png'.format(self.method, self.name))
        ### END - if alg == all

        if alg in ('DBSCAN', 'ward', 'average', 'complete'):
            plt.figure()
            if alg == 'DBSCAN':
                label = self.db[0]
            else:
                label = self.agglo[0]

            plt.scatter(self.X[:, 0], self.X[:, 1], c=label, cmap=plt.cm.spectral)
            plt.title('{0} of {1}'.format(alg, self.name))
            plt.savefig('{0}_{1}_{2}.png'.format(alg, self.method, self.name))
            ### END - if alg in
    ### END - def plotCluster


    def evaluate(self, labels):
        result = tuple()
        for label in labels:
            homo = metrics.homogeneity_score(self.y, label)
            comp = metrics.completeness_score(self.y, label)
            vmes = metrics.v_measure_score(self.y, label)
            ari = metrics.adjusted_rand_score(self.y, label)
            result = result + (homo, comp, vmes, ari)
        ### END - for label
        return result
    ### END - def evaluate
### END - class Clustering