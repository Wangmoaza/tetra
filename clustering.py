from sklearn.cluster import *
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from supplementary import *


class Clustering:
    def __init__(self, name, method, X, y, n_clusters=None):
        self.name = name
        self.method = method
        self.X = X
        self.db = None  # predicted label in tuple
        self.agglo = None  # predicted label in tuple order 'ward', 'average', 'complete'
        
        if y is None:
            self.y = np.zeros(shape=self.X.shape[0], dtype=int)
        else:
            self.y = y
        
        if n_clusters is None:
            self.n_clusters = np.unique(self.y).shape[0]
        else:
            self.n_clusters = n_clusters
    ### END - def __init__


    def dbscan(self, min_samples):
        model = DBSCAN(eps=3, min_samples=min_samples)
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
            knn_graph = kneighbors_graph(self.X, 10)
        else:
            knn_graph = None

        if linkage in ('ward', 'average', 'complete'):
            
            model = AgglomerativeClustering(linkage=linkage,
                                            n_clusters=self.n_clusters,
                                            connectivity=knn_graph)
            model.fit(self.X)
            self.agglo = (model.labels_,)
        ### END - if linkage

        elif linkage == 'all':
            label_list = []
            for link in ('ward', 'average', 'complete'):
                model = AgglomerativeClustering(linkage=link,
                                                n_clusters=self.n_clusters,
                                                connectivity=knn_graph)
                print link
                print self.X.shape
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


    def plotCluster(self, alg='all', num=None):
        titles = ['True Clusters', 'DBSCAN', 'Ward', 'Average', 'Complete']
        labels = [self.y, self.db[0]]
        
        for lab in self.agglo:
            labels.append(lab)

        # plot all clustering alg
        if alg == 'all':
            plt.figure()
            for i, title, label in zip(range(len(titles)), titles, labels):
                plt.subplot(2, 3, i + 1)
                plt.scatter(self.X[:, 0], self.X[:, 1], c=label, cmap=plt.cm.spectral, s=10)
                plt.title(title, fontdict=dict(verticalalignment='top'))
                # FIXME
                #plt.axis('equal')
                #plt.autoscale()
                plt.axis('off')
            ### END - for i, title, label

            plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
            plt.suptitle('Clustering of {0}'.format(self.name), size=17)
            if self.X.shape[1] == 3:
                if num is None:
                    plt.savefig('Clustering_{0}_{1}.png'.format(self.method, self.name))
                else:
                    plt.savefig('Clustering_{0}_{1}_{2}.png'.format(self.method, self.name, num))
            elif self.X.shape[1] == 3:
                print "Error: 3d not supported"
        ### END - if alg == all

        # plot single clustering alg
        if alg in ('DBSCAN', 'ward', 'average', 'complete'):
            plt.figure()
            if alg == 'DBSCAN':
                label = self.db[0]
            else:
                label = self.agglo[0]

            if self.X.shape[1] == 2: # 2-dimensional
                plt.scatter(self.X[:, 0], self.X[:, 1], c=label, cmap=plt.cm.spectral, s=20)
                plt.title('{0} of {1}'.format(alg, self.name))
                if num is None:
                    plt.savefig('{0}_{1}_{2}.png'.format(alg, self.method, self.name))
                else:
                    plt.savefig('{0}_{1}_{2}_{3}.png'.format(alg, self.method, self.name, num))
            ### END - self.X.shape[1] == 2
            
            elif self.X.shape[1] == 3: # 3-dimensional
                title = "{0} of {1}".format(alg, self.name)
                figName = '{0}_{1}_{2}'.format(alg, self.method, self.name)
                if num is not None:
                    figName += '_{0}'.format(num)
                plot3d(self.X, label, title, figName)
            ### END - self.X.shape[1] == 3:
        ### END - if alg in
    ### END - def plotCluster


    def evaluate(self, labels):
        result = tuple()
        # true_label provided
        if self.n_clusters is None:
            for label in labels:
                ami = metrics.adjusted_mutual_info_score(self.y, label)
                nmi = metrics.normalized_mutual_info_score(self.y, label)
                vmes = metrics.v_measure_score(self.y, label)
                ari = metrics.adjusted_rand_score(self.y, label)
                result = result + (ami, nmi, vmes, ari)
            ### END - for label
        ### END - if self.y

        else:
            for label in labels:
                result = result + (metrics.silhouette_score(self.X, label), )
        return result
    ### END - def evaluate
    
    def pred_labels(self, alg):
        if (alg == 'DBSCAN') and (self.db is not None):
            return self.db[0]
        
        # FIXME
        # 'average' and 'complete' may not be in that position
        elif alg == 'ward':
            try:
                return self.agglo[0]
            except:
                pass
        elif alg == 'average':
            try:
                return self.agglo[1]
            except:
                pass
        elif alg == 'complete':
            try:
                return self.agglo[2]
            except:
                pass
        else:
            print "Error: invalid parameter"
            return None
    ### END - def pred_labels
    
### END - class Clustering