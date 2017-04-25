from sklearn.cluster import *
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
from scipy.cluster.hierarchy import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from supplementary import *
from scipy.spatial.distance import *
from ete3 import ClusterTree, TreeStyle
from itertools import combinations

class Clustering:
    """ This class uses scikit-learn implementation for clustering.
    
    Able to conduct agglomerative clustering (ward, average, complete) and DBSCAN
    """
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
                plt.scatter(self.X[:, 0], self.X[:, 1],
                            c=label, cmap=plt.cm.spectral, s=10)
                plt.title(title, fontdict=dict(verticalalignment='top'))
                # FIXME
                # plt.axis('equal')
                # plt.autoscale()
                plt.axis('off')
            ### END - for i, title, label

            plt.subplots_adjust(left=0.125, right=0.9,
                                bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
            plt.suptitle('Clustering of {0}'.format(self.name), size=17)
            if self.X.shape[1] == 3:
                if num is None:
                    plt.savefig('Clustering_{0}_{1}.png'.format(
                        self.method, self.name))
                else:
                    plt.savefig('Clustering_{0}_{1}_{2}.png'.format(
                        self.method, self.name, num))
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

            if self.X.shape[1] == 2:  # 2-dimensional
                plt.scatter(self.X[:, 0], self.X[:, 1],
                            c=label, cmap=plt.cm.spectral, s=20)
                plt.title('{0} of {1}'.format(alg, self.name))
                if num is None:
                    plt.savefig('{0}_{1}_{2}.png'.format(
                        alg, self.method, self.name))
                else:
                    plt.savefig('{0}_{1}_{2}_{3}.png'.format(
                        alg, self.method, self.name, num))
            ### END - self.X.shape[1] == 2

            elif self.X.shape[1] == 3:  # 3-dimensional
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


class Scipy_Clustering:
    """ This class uses scipy implementation to perform hierarchical clustering."""

    def __init__(self, X, y=None, name='name', method='method', ids=None, n_clusters=None):
        """
        Args:
            name (str): taxon name of the target matrix
            method (str): dimensionality reduction method
            X (ndarray): [n_samples, n_features] matrix
            y (ndarray): true labels of n_samples if exists
            n_clusters (int): number of clusters if specified
            ids (ndarray or list): list of NCBI project access ids
        """
        self.name        = name
        self.method      = method
        self.ids         = ids # NCBI project access id list
        self.X           = X
        self.Z           = None # linkage matrix
        self.alg         = None # linkage algorithm
        self.pred_y      = None # predicted labels
        if y is None:
            # set all index to zero if not given
            self.y       = np.zeros(shape=self.X.shape[0], dtype=int)
        else:
            self.y       = y

        if n_clusters is None:
            self.n_clusters = np.unique(self.y).shape[0]
        else:
            self.n_clusters = n_clusters
    ### END - def __init__

    def set_n_clusters(self, n_clusters):
        """ sets new n_clusters
        Args:
            n_clusters (int): number of clusters
        """
        self.n_clusters = n_clusters;
    ### END - def set_n_cluster

    def plot_cluster(self, num=None):
        """ Plot clustered result of data on 2d plane"""
         plt.figure()
         alg = 'Ward-scipy'
        if self.X.shape[1] == 2:  # 2-dimensional
            plt.scatter(self.X[:, 0], self.X[:, 1],
                        c=self.pred_y, cmap='prism', s=20)
            plt.title('{0} of {1}'.format('Ward Clustering', self.name))
            if num is None:
                plt.savefig('{0}_{1}_{2}.png'.format(
                    alg, self.method, self.name))
            else:
                plt.savefig('{0}_{1}_{2}_{3}.png'.format(
                    alg, self.method, self.name, num))
        ### END - self.X.shape[1] == 2
        elif self.X.shape[1] == 3:  # 3-dimensional
            print("Not Implemented")
        ### END - self.X.shape[1] == 3:

        plt.show()
    ### END - def plot_cluster

    def plot_dendrogram(self *args, **kwargs):
        """ plot and save dendrogram from clustered data using __fancy_dendrogram """
        plt.figure()
        self.__fancy_dendrogram(*args, **kwargs)
        plt.show()
        plt.savefig('{0}_{1}_{2}.png'.format('dendrogram', self.method, self.name))
    ### END - def plot_dendrogram

    def __fancy_dendrogram(self, *args, **kwargs):
        """ private method for plot_dendrogram
        modified from https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
        Args:
        """
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0) 

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            if not kwargs.pop('truncate_mode', None): # truncate_mode ON
                plt.title('HC Dendrogram of {0} (truncated)'.format(self.name))
            else:
                plt.title('HC Dendrogram of {0}'.format(self.name))

            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
                ### END - if y
            ### END - for i, d, c

            if max_d:
                plt.axhline(y=max_d, c='k')
        ### END - if not
        
        return ddata
    ### END - fancy dendrogram

    def dendrogram_to_newick(self):
        """ Convert Dendrogram to newick format. """
        # FIXME
        # how do I make leaf_names (lisft with names of leaves)
        tree = to_tree(self.Z)
        self.__getNewick(tree, "", tree.dist, self.ids)
    ### END - dendrogram_to_newick

    def __getNewick(self, node, newick, parentdist, leaf_names):
        """ private method for dendrogram_to_newick.
        modified from http://stackoverflow.com/questions/28222179/save-dendrogram-to-newick-format
        
        Args:
            node (ClusterNode object): tree
            newick (str): tranformed newick format
            parentdist:
            leaf_names
        
        Returns:
            newick
        """
        if node.is_leaf():
            return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
        else:
            if len(newick) > 0:
                newick = "):%.2f%s" % (parentdist - node.dist, newick)
            else:
                newick = ");"
            newick = self.__getNewick(node.get_left(), newick, node.dist, leaf_names)
            newick = self.__getNewick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
            newick = "(%s" % (newick)
        
        return newick
    ### END - def __getNewick

    def cluster(alg='ward', metric='euclidean'):
        """ performs hierarchical clustering.
        Args:
            alg (string): linkage algorithm
            metric (string): metric

        Returns:
            predicted labels of the datapoints (ndarray [n_sampes])
        """
        self.Z = linkage(self.X, alg) # generate linkage matrix
        
        # calculate cophnenetic correlation coefficient
        c, coph_dists = cophenet(self.Z, pdist(self.X))
        print "cophenetic correlation coefficient: {0}".format(c)

        if self.n_clusters == 0:
            self.pred_y = fcluster(self.Z) # use default inconsistency method
        else
            self.pred_y = fclusters(self.Z, t=self.n_clusters, criterion='maxclust')

        return self.pred_y
    ### END - def cluster
### END - class Scipy_Clustering

def newick_to_linkage(filePath):
    """ converts newick tree to scipy linkage matrix """
    tree                   = ClusterTree(filePath)
    leaves                 = tree.get_leaf_names()
    ts                     = TreeStyle()
    ts.show_leaf_name      = True
    ts.show_branch_length  = True
    ts.show_branch_support = True

    idx_dict = {}
    idx = 0
    for leaf in leaves:
        idx_dict[leaf] = idx
        idx += 1

    idx_labels = [idx_dict.keys()[idx_dict.values().index(i)] for i in range(len(idx_dict))]

    dmat = np.zeros((len(leaves), len(leaves))) # FIXME need to understand

    for leaf1, leaf2 in combinations(leaves, 2):
        d = tree.get_distance(leaf1, leaf2)
        dmat[idx_dict[leaf1], idx_dict[leaf2]] = dmat[idx_dict[leaf2], idx_dict[leaf1]] = d

    schlink = sch.linkage(scipy.spatial.distance.squareform(dmat),method='average',metric='euclidean')

