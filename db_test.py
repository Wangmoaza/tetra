import numpy as np

def main():
    # load dataset
    phylums = np.load('db2_phylums.npy')
    taxons = np.load('db2_taxons.npy')
    ids = np.load('db2_ids.npy')
    tetra = np.load('db2_tetra.npy')
  
    # make dictionary {phylum: count}
    dic = {}
    for phy in taxons[:, 1]:
        dic[phy] = dic.get(phy, 0) + 1
    
    phylumNames = [] 
    sum = 0
    for phy in dic.keys():
        if dic.get(phy) >= 500:
            sum += dic.get(phy)
            phylumNames.append(phy)
            print (phy, dic.get(phy))
    #phylumNames.append('Others')
    print phylumNames
    np.save('db2_phylumNames', np.asarray(phylumNames))
    print sum
        
main()
        
