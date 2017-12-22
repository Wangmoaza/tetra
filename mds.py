from sklearn.manifold import MDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from clustering import *
from divide_db_level import divide
from supplementary import *
import time

ranks = ["domain", "phylum", "class", "order", "family", "genus", "species"]

def perform(rank_num, group, minCnt=30, dim=2, tsne=False):
    method = "MDS"
    X, y, names = load_data(tetraFile='db2_tetra_{0}_{1}.npy'.format(ranks[rank_num], group),
                            taxonFile='db2_taxon_{0}_{1}.npy'.format(ranks[rank_num], group), 
                            rank=rank_num + 1, minCnt=minCnt)
    
    if X.shape[0] == 0:
        print "Error: sample size is 0"
        return

    mds = MDS(n_components=2, n_init=1, max_iter=100)
    X_2d = mds.fit_transform(X)
    title = "MDS of TNA of " + group
    figName = "MDS_{0}_{1}".format(ranks[rank_num][0], group)

    plot2d(X_2d, y, names, title, figName)

    print '...clustering'
    name = '{0}_{1}'.format(ranks[rank_num][0], group)
    clust = Clustering(name, method, X_2d, y)
    #d_ami, d_nmi, d_vmes, d_ari = clust.dbscan(5)
    eval_tuple = clust.agglomerative(linkage='ward', connect=True)
    clust.plotCluster(alg='ward')

    with open('result_score.txt', 'a') as f:
        mse_result = '-'
        general_str = "{rank}\t{group}\t{method}\t{size}\t{n_cluster}\t{mse}\t{dim}\t".format(rank=ranks[rank_num],
                                                                        group=group, method=method,                                                                                     n_cluster=np.unique(y).shape[0],
                                                                        dim=dim, size=X_2d.shape[0],
                                                                        mse=mse_result)
        #d_values_str = "{ami:0.3f}\t{nmi:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\t".format(ami=d_ami, nmi=d_nmi,
        #                                                                            vmes=d_vmes, ari=d_ari)
        #line = general_str + d_values_str
        line = general_str
        
        for i in range(len(eval_tuple) / 4):
            a_ami, a_nmi, a_vmes, a_ari = eval_tuple[i], eval_tuple[i+1], eval_tuple[i+2], eval_tuple[i+3]
            a_values_str = "{ami:0.3f}\t{nmi:0.3f}\t{vmes:0.3f}\t{ari:0.3f}\t".format(ami=a_ami, nmi=a_nmi,
                                                                                        vmes=a_vmes, ari=a_ari)
            line += a_values_str
        ### END - for i
        line += '\n'
        f.write(line)
    ### END - with open

def main():
    flag = input("1 for specific, 2 for all under 3 for etc: ")
    
    if flag == 2:
        # loop from phylum to family
        for i in range(2, len(ranks) - 2):
        #for i in range(1, 2):
            # generate taxon and tetra file for rank under this rank
            # e.g. if current rank is class, then generate datasets for order
            names = np.load('db2_{0}_names.npy'.format(ranks[i]))  # e.g. ranks[i] == class

            # peform pca
            for group in names:
                print
                '\n'
                print
                '******** ' + group + ' ********'
                perform(i, group)
                ### END - for group
                ### END - for i
    ### END - if flag == 2

    elif flag == 1:  # specific
        rankName = input("taxonomical rank: ")
        taxonName = input("taxon name: ")
        perform(ranks.index(rankName), taxonName, minCnt=30)
    ### END - elif flag == 1

    elif flag == 3:
        genus_list = np.load('db2_genus_list.npy')
        rank_num = ranks.index('genus')
        
        t0 = time.time()
        for group in genus_list:
            #perform(rank_num, group, minCnt=10, dim=2, tsne=False)
            perform(rank_num, group, minCnt=10, dim=2, tsne=True)
        ### END - for group
        print "runtime ", time.time() - t0
    ### END - elif flag == 3
### END - main

main()