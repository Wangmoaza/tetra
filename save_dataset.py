import numpy as np
import random

def parse(filePath):
    IDs = []
    taxons = []
    phylums = []
    tetra = []
    with open(filePath) as file:
        for line in file.readlines():
            tokens = line.strip().split('\t')
            IDs.append(tokens[0])
            taxons.append([i for i in tokens[1].split(';')[:7]])
            tetra.append([float(i) for i in tokens[2].split(',')])
        ### END - for line
    ### END - with
    IDs = np.asarray(IDs)
    taxons = np.asarray(taxons)
    tetra = np.asarray(tetra)
    print IDs.shape
    print taxons.shape
    print tetra.shape
    return IDs, taxons, tetra
### END - def parse

def getPhylumIndex(taxons, phylumNames):
    # make list of phylum names in index 0: Bacteriodetes, etc.
    phylums = taxons[:, 1]
    indexList = []
    
    for i in range(len(phylums)):
        phy = phylums[i]
        if phy in phylumNames:
            indexList.append(phylumNames.index(phy))
        else:
            indexList.append(len(phylumNames)-1)
    ### END - for i
        
    for i in range(len(phylumNames)):
        print i, indexList.count(i)
    
    return np.asarray(indexList)
### END - def getPhylumIndex

def randSelect(phylumIndex, phylumNames, size=500):
    """ randomly selects the same size of samples from each phylum
    
        param:
            phylumIndex
            phylumNames
            size=500
        return:
            selectedIndices
    """
    
    basketList = [ list() for i in range(len(phylumNames)) ]
    print len(phylumIndex)
                                                                                                                                                 
    # put indices in each basket
    for j in range(len(phylumIndex)):
        basketList[ phylumIndex[j] ].append(j)
    
    # select n-size items for each basket
    selectedIndices= []
    for i in range(len(basketList)):
        random.shuffle(basketList[i])
        
        for idx in basketList[i][:size]:
            selectedIndices.append(idx)
    ### END - for i
    
    selectedIndices = sorted(selectedIndices)
    return selectedIndices
    
    #newIDs = IDs[selectedIndices]
    #newTaxons = taxons[selectedIndices, :]
    #newTetra = tetra[selectedIndices, :]
    #newPhylumIndex = phylumIndex[selectedIndices]
    
    #np.save('db2_selected_ids', newIDs)
    #np.save('db2_selected_taxons', newTaxons)
    #np.save('db2_selected_tetra', newTetra)
    #np.save('db2_selected_phylumIndex', newPhylumIndex)
### END - def randSelect
        
        
def saveTop(IDs, taxons, tetra, phylumIndex, phylumNames, includeOthers=True):
    # indices of data that are in phylumNames
    indices = []
    for i in range(len(phylumIndex)):
        phy = phylumIndex[i]
        if includeOthers == True:
            if phy in range(len(phylumNames)-1): # exclude others
                indices.append(i)
        else:
            if phy in range(len(phylumNames)):
                indices.append(i)
    ### END - for i

    newIDs = []
    newTaxons = []
    newTetra = []
    newPhylumIndex = []
    
    newIDs = IDs[indices]
    newTaxons = taxons[indices, :]
    newTetra = tetra[indices, :]
    newPhylumIndex = phylumIndex[indices]

    print "***** new *****"
    print newIDs.shape
    print newTaxons.shape
    print newTetra.shape
    print newPhylumIndex.shape


    np.save('db2_ids_top', newIDs)
    np.save('db2_taxons_top', newTaxons)
    np.save('db2_tetra_top', newTetra)
    np.save('db2_phylums_top', newPhylumIndex)
    return indices
### END - saveTop

def getTopPhylumList(taxons, minCnt=500, includeOthers=True):
    # make dictionary {phylumName: count}
    dic = {}
    for phy in taxons[:, 1]:
        dic[phy] = dic.get(phy, 0) + 1

    phylumNames = []
    total = 0
    for phy in dic.keys():
        if dic.get(phy) >= minCnt:
            total += dic.get(phy)
            phylumNames.append(phy)
            print (phy, dic.get(phy))
    ### END - for phy
    print "total", total
    
    if includeOthers:
        phylumNames.append('Others')
    
    print phylumNames
    
    np.save('db2_phylumNames', np.asarray(phylumNames))
    return phylumNames
### END - def getTopPhylumList

def main():
    Ids, taxons, tetra = parse("tetra.txt")
    phylumNames = getTopPhylumList(taxons, minCnt=1000)
    phylumIndex = getPhylumIndex(taxons, phylumNames)
    saveTop(Ids, taxons, tetra, phylumIndex, phylumNames)
    np.save('db2_tetra', tetra)
    np.save('db2_taxons', taxons)
    np.save('db2_ids', Ids)
    np.save('db2_phylums', phylumIndex)
