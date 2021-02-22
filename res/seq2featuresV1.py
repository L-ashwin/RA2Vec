import numpy as np
import pandas as pd
import gensim
import re
from itertools import product

class Transformer(object):
    """docstring for ."""

    def __init__(self):
        
        self.transCodes = pd.read_csv('./model/data/mapping.csv')

        self.W2V_models = None
        self.GappedPairFreq = None
        self.ProtVecFlag = False

        self.xData = None
        self.yData = None

    def set_modelList(self, W2V_models, ProtVec=None, GappedPairFreq=False):

        self.ProtVec = ProtVec
        self.GappedPairFreq = GappedPairFreq
        
        if isinstance(W2V_models, list):
            self.W2V_models = W2V_models.copy()
        else:
            self.W2V_models = [W2V_models]

        if self.ProtVec is not None:
            self.W2V_models.append(self.ProtVec)
            self.ProtVecFlag = True

    def _get_transDict(self, trans):
        if trans == 'ProtVec':
            return None

        amino = self.transCodes
        dic = {}
        for i in range(amino.shape[0]):
            dic[ord(amino['one_letter_code'][i])] = ord(amino[trans][i])
        return dic

    def get_seq2vec(self, seq, w2vModel_gen, transDict, kGrams):
        if transDict is not None:
            seq = seq.translate(transDict)

        gram2vec = []
        for i in range(0, len(seq) - (kGrams - 1)):
            try:
                gram2vec.append(w2vModel_gen.wv.__getitem__(seq[i:i + kGrams]))
            except:
                continue
                # print('word not in dictionary', seq[i:i+kGrams])

        if gram2vec == []:
            seq2vec = np.zeros(w2vModel_gen.vector_size)
            print(seq, 'set to all Zeros')
        else:
            seq2vec = np.sum(gram2vec, axis=0)

        return seq2vec

    def get_embed(self, data, w2vModel_gen, transDict, kGrams):
        xData = []
        for seq in data:
            seq2vec = self.get_seq2vec(seq, w2vModel_gen, transDict, kGrams)
            xData.append(seq2vec)
        return np.array(xData)

    def _gen_pairGapFreq(self, seq, gap_size, transDict):
        if transDict is not None:
            seq = seq.translate(transDict)

            _alphabet_ = list(map(chr, list(set(transDict.values()))))
            pairs = product(_alphabet_, repeat=2)
            pairs = list(map(''.join, pairs))

        else:
            pairs = product(
                ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'],
                repeat=2)
            pairs = list(map(''.join, pairs))

        amino_pair_counts_ = dict.fromkeys(pairs, np.float32(0))

        for i, j in zip(range(0, len(seq)), range(gap_size + 1, len(seq))):
            pair = seq[i] + seq[j]
            try:
                amino_pair_counts_[pair] = amino_pair_counts_[pair] + np.float32(1)
            except:
                pass

        return np.array(list(amino_pair_counts_.values()), 'float32')

    def get_pairGapFreq(self, data, gap_size, transDict):
        X_gf = []
        for seq in data:
            x_gf = self._gen_pairGapFreq(seq, gap_size, transDict)
            X_gf.append(x_gf)
        return np.array(X_gf)

    def set_data(self, data, target):
        xData = []
        for model in self.W2V_models:
            w2vModel_gen = gensim.models.Word2Vec.load(model.location)
            kGrams = model.kGram
            translation = model.Model

            transDict = self._get_transDict(translation)

            X = self.get_embed(data, w2vModel_gen, transDict, kGrams)
            xData.append(X)
        xData = np.concatenate(xData, 1)

        if self.GappedPairFreq:
            X_gf = self.get_pairGapFreq(data, 3, None)
            xData = np.concatenate((xData, X_gf), 1)
        yData = target

        self.xData = xData
        self.yData = target

class W2V_Model(object):
    def __init__(self, loc=None):
        self.location = None
        self.Model    = None
        self.kGram    = None
        self.window   = None
        self.vecSize  = None

        if loc is not None:
            self.set_attributes_byName(loc)

    def set_attributes_byName(self, loc):
        import re
        self.location = loc
        self.Model    = re.findall('RA2V_(\S+)_G', loc)[0]
        self.kGram    = int(re.findall('_G([0-9]+)', loc)[0])
        self.window   = int(re.findall('_W([0-9]+)', loc)[0])
        self.vecSize  = int(re.findall('_S([0-9]+)', loc)[0])

    def __str__(self):
        return f'{self.Model}_G{self.kGram}_S{self.vecSize}_W{self.window}'
    
class GetModels():
    @staticmethod
    def singles(model_loc, modelComb):
        import os
        from res.runBuilder import RunBuilder
        models = os.listdir(model_loc)
        models = [model_loc+i for i in models]
        models.sort()
    
        modelParams = RunBuilder.get_runs(modelComb)
        W2V_models = []
        for param in modelParams:
            string = f'{param.alphabet}_G{param.kGram}_S{param.vecSize}_W{param.window}.model'
            model = None
            for address in models:
                if address.endswith(string):# string in address:
                    model = W2V_Model()
                    model.set_attributes_byName(address)

            if model is None:
                print(string, '.....Pattern Not Found!!')
            else:
                W2V_models.append(model)
                
        return W2V_models
