import numpy as np
import random
class KmerCounter:
    def __init__(self, filePath):
        self.seqs, self.ids, self.taxonomy = self.getInput(filePath)
        self.counts = []
        self.frequencies = []
        self.phylumNames = ['Bacteroidetes', 'Acidobacteria', 'Proteobacteria', 'Firmicutes', 'Actinobacteria', 'Planctomycetes', 'Chloroflexi', 'Others']
    def getInput(self, filePath):
        seqs = []
        ids = []
        taxonomy = []
        with open(filePath, "r") as file:
            file.readline()
            for line in file.readlines():
                tokens = line.strip().split(",")
                ids.append(tokens[0].replace('"', ''))
                taxonomy.append(tokens[7].replace('"', '').split(';')[:2])
                seqs.append(tokens[6].replace('"', ''))
            ### END - for line
        ### END - with
        return seqs, ids, taxonomy
    ### END - def getInput

    def count_kmer(self, k):
        indexDict = make_dict(make_kmer_list([''], k))
        cnt = 0
        for seq in self.seqs:
            cntList = [0] * 256
            #seq = self.changeNts(seq) # change other nts to A, G, C, T

            for i in range(len(seq) - k + 1):
                kmer = seq[i:i + k]
                try:
                    cntList[ indexDict[kmer] ] += 1
                except KeyError:
                    pass
                
                rev_kmer = reverse_complement(kmer)
                try:
                    cntList[ indexDict[rev_kmer] ] += 1
                except KeyError:
                    pass
            ### END - for i
            self.counts.append(cntList)
            cnt += 1
            #if cnt % 100 == 0:
            #    print "finished seq ",  cnt
        ### END - for seq
        self.counts = np.asarray(self.counts)
    ### END - def count_kmer

    def cal_kmer_freq(self, k):
        self.count_kmer(k)
        self.frequencies = np.zeros(self.counts.shape)
        summed = np.sum(self.counts, axis=1)
        for row in range(self.counts.shape[0]): # number of samples
            for column in range(self.counts.shape[1]): # number of tetranucleotides
                self.frequencies[row, column] = float(self.counts[row, column]) / summed[row]

        return self.frequencies
    ### END - def cal_kmer_freq

    def getIDs(self):
        return self.ids

    def getTaxonomy(self):
        return self.taxonomy

    def getPhylumIndex(self):
        taxonomy = np.asarray(self.taxonomy)
        print taxonomy.shape
        print taxonomy
        phylums = taxonomy[:, 1]
        indexList = []

        for i in range(len(phylums)):
            phy = phylums[i]
            if phy in self.phylumNames:
                indexList.append(self.phylumNames.index(phy))
            else:
                indexList.append(len(self.phylumNames)-1)
        
        return indexList

    def toFile(self):
        np.save('tetrafreq', self.cal_kmer_freq(4))
        np.save('phylumNames', self.phylumNames)
        np.save('phylumIndex', self.getPhylumIndex())

    def count(self, nt):
        cnt = 0
        for seq in self.seqs:
            cnt += seq.count(nt)
        return cnt

    def totalLength(self):
        cnt = 0
        for seq in self.seqs:
            cnt += len(seq)
        return cnt

    def changeNts(self, seq):
        # change nucleotides that are not A, G, C, T
        ntList = list(seq)
        for i in range(len(ntList)):
            nt = ntList[i]
            if nt == 'R':
                ntList[i] = random.choice(['A', 'G'])
            elif nt == 'Y':
                ntList[i] = random.choice(['C', 'T'])
            elif nt == 'N':
                ntList[i] = random.choice(['A', 'T', 'G', 'C'])
            elif nt == 'W':
                ntList[i] = random.choice(['A', 'T'])
            elif nt == 'S':
                ntList[i] = random.choice(['G', 'C'])
            elif nt == 'M':
                ntList[i] = random.choice(['A', 'C'])
            elif nt == 'K':
                ntList[i] = random.choice(['T', 'G'])
            elif nt == 'B':
                ntList[i] = random.choice(['T', 'G', 'C'])
            elif nt == 'H':
                ntList[i] = random.choice(['A', 'T', 'C'])
            elif nt == 'D':
                ntList[i] = random.choice(['A', 'T', 'G'])
            elif nt == 'V':
                ntList[i] = random.choice(['A', 'G', 'C'])
        ### END - for i
        newSeq = ''.join(ntList)
        return newSeq
    ### END - def changeNts
### END - class

def make_kmer_list(myList, k):
    newList = []
    if k == len(myList[0]):
        return myList

    for item in myList:
        for nt in ['A', 'C', 'G', 'T']:
            newList.append(item + nt)

    return make_kmer_list(newList, k)
### END - def make_kmer_list

def make_dict(kmerList):
    kmerDict = {}
    for index in range(len(kmerList)):
        kmerDict[kmerList[index]] = index

    return kmerDict
### END - def make_dict

def reverse_complement(seq):
    comp_seq = ""
    for ch in seq:
        comp_seq = complementary_base(ch) + comp_seq
    return comp_seq
### END - def reverse_complement

def complementary_base(nt):
    if nt == 'A':
        return 'T'
    elif nt == 'T':
        return 'A'
    elif nt == 'G':
        return 'C'
    elif nt == 'C':
        return 'G'
    else:
        #print "invalid nucleotide"
        return 'N'
### END - def complementary_base
def main():
    counter = KmerCounter('../autoencoder/seq.csv')
    counter.toFile()

main()
