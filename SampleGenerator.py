import numpy as np
from save_dataset import *

class SampleGenerator:
    def __init__(idFile='db2_ids.npy',
                taxonFile='db2_taxons.npy',
                tetraFile='db2_tetra.npy',
                phylumFile='db2_phylums.npy',
                labelFile='db2_phylumNames.npy'):
        
        self.IDs = np.load(idFile)
        self.taxons = np.load(taxonFile)
        self.tetra = np.load(tetraFile)
        self.phylums = np.load(phylumFile)
        self.labelNames = np.load(labelFile)
        self.sampleIndices = []
    
    def generate(self, sizePerClass=500):
        self.sampleIndices = randSelect(self.phylums, self.labels, size=sizePerClass)
        newTetra = self.tetra[self.sampleIndices, :]
        newPhylumIndex = self.phylums[self.sampleIndices]
        return newTetra, newPhylumIndex
    
    def getSampleIndices(self):
        return self.sampleIndices
    
    def getLabelNames(self):
        return self.labelNames
    
    def getAllData(self):
        """ get all datasets 
            return:
                newIDs
                newTaxons
                newTetra
                newPhylums
        """
        newIDs = self.IDs[self.sampleIndices]
        newTaxons = self.taxons[self.sampleIndices, :]
        newTetra = self.tetra[self.sampleIndices, :]
        newPhylums = self.phylums[self.sampleIndices]
        
        return newIDs, newTaxons, newTetra, newPhylums
        
    