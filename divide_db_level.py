import numpy as np
from collections import defaultdict

def divide(level, taxonFile, tetraFile, lowerbound):
    taxons = np.load(taxonFile) # 2d (sample * taxon)
    tetra = np.load(tetraFile) # 2d (sample * 256 tetra freq)
    ranks = ["domain", "phylum", "class", "order", "family", "genus", "species"]
    
    countdic = {}
    tetradic = defaultdict(list)
    taxondic = defaultdict(list)
    nameList = []
    i = 0
    
    # count samples per each label
    for tax in taxons:
        countdic[tax[level]] = countdic.get(tax[level], 0) + 1
    
    # make list of names that contains samples more than lowerbound
    for key in countdic.keys():
        if countdic[key] >= lowerbound:
            nameList.append(key)
    
    nameList.append('Others')
    
    for tax in taxons:
        if tax[level] in nameList:
            tetradic[tax[level]].append(tetra[i, :])
            taxondic[tax[level]].append(tax)
        else:
            tetradic['Others'].append(tetra[i, :])
            taxondic['Others'].append(tax)
        i += 1
    ### END - for tax
    
    idx = 0
    taxList = []
    
    # save new datasets
    for key in tetradic.keys():
        taxName = 'db2_taxon_' + ranks[level] + "_" + key
        tetraName = 'db2_tetra_' + ranks[level] + "_" + key
        
        np.save(taxName, taxondic[key])
        np.save(tetraName, tetradic[key])
    
    labelName = 'db2_' + ranks[level] + "_names"
    np.save(labelName, nameList)
        
    ### END - for key
### END - def divide
                 
def main():
    taxonFile = 'db2_taxons_top.npy'
    tetraFile = 'db2_tetra_top.npy'
    divide(1, taxonFile, tetraFile, 500)

main()

        