import logging
import torch
import torch.utils.data as data
import numpy as np
import json
import os
import gzip
import tqdm

from collections import OrderedDict
import os
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample,LoggingHandler
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sentence_transformers import models, losses,util,datasets,evaluation
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdMolDescriptors
# from rdkit import DataStructs
# feature_extractor = AutoFeatureExtractor.from_pretrained("newoutputs/checkpoint-76582")

nbits = 1024#1024
longbits = 16384
# RDKit descriptors -->
calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
# dictionary
fpFunc_dict = {}
fpFunc_dict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
fpFunc_dict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
fpFunc_dict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
fpFunc_dict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
fpFunc_dict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
fpFunc_dict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
fpFunc_dict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
fpFunc_dict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
fpFunc_dict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
fpFunc_dict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=longbits)
fpFunc_dict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=longbits)
#fpFunc_dict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
fpFunc_dict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
fpFunc_dict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits)
#fpFunc_dict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
#fpFunc_dict['laval'] = lambda m: fpAvalon.GetAvalonFP(m, longbits)
fpFunc_dict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)
fpFunc_dict['rdkDes'] = lambda m: calc.CalcDescriptors(m)

class SynergyEncoderDataset(data.Dataset):
    def __init__(self, drug_1_smiles,drug_2_smiles,Y_Label,context,maxCompoundLen=128, device=torch.device("cuda"),fp_name='hashap'):#context:细胞系，Y：LABEL，二分类，这里的maxcompoundlen是相当于batch-size吗？device="cpu"
        self.maxCompoundLen = maxCompoundLen
        self.drug_1_smiles = drug_1_smiles
        self.drug_2_smiles = drug_2_smiles
        self.context = context
        self.Y = Y_Label
        self.device = device
        self.fp_name = fp_name
        self.len = len(self.Y)#具体有多少数据
        self.features = json.loads(open("./drugcombdb/context_set_m.json", 'r').read())
        self.drug_set = json.loads(open("./drugcombdb/drug_set.json", 'r').read())
#./drugcomb/drug-set.json
        # word_embedding_model = models.Transformer("DeepChem/ChemBERTa-77M-MLM", max_seq_length=128)
        # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        # self.drug_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        model_name = 'output/simcsesqrt-model'
        self.drug_model = SentenceTransformer(model_name,device=device)
        self.encode_smiles()

    def __len__( self ):
        return self.len
    
    def encode_smiles(self):
        self.simse = {}
        # self.chem ={}
        self.drug2fps = {}
        for smile in list(set(self.drug_1_smiles))+list(set(self.drug_2_smiles)):   
            smile_str = self.drug_set[smile]['smiles'][:self.maxCompoundLen]
            self.simse[smile] = self.drug_model.encode(smile_str)
            # self.chem[smile] = self.drug_model.encode(smile_str)
            mol = Chem.MolFromSmiles(self.drug_set[smile]['smiles'])
            fp = fpFunc_dict[self.fp_name](mol)
            self.drug2fps[smile] = np.asarray(fp)
        # train_set = list(set(train_drug_2_cv))+list(set(train_drug_1_cv))
        # train_set = train_data['drug_1'].astype(str).to_list() + train_data['drug_2'].astype(str).to_list()
        # mols_train = [Chem.MolFromSmiles(self.drug_set[s]['smiles']) for s in train_set]
        # fp_train = [fpFunc_dict[self.fp_name](mols) for mols in mols_train]
        # l = list(set(test_drug_2_cv)) + list(set(test_drug_1_cv))
        # mols_test = [Chem.MolFromSmiles(self.drug_set[d]['smiles']) for d in l]
        # fp_test = [fpFunc_dict[self.fp_name](mols) for mols in mols_test]
        # from rdkit import DataStructs
        # x = 0
        # for i in fp_train:
        #     for j in fp_test:
        #         ds_1 = DataStructs.FingerprintSimilarity(i, j)
        #         x = x + ds_1
        #         t = x / (len(fp_test) * len(fp_train))
        # tn_1 = DataStructs.TanimotoSimilarity(i, j)
        # x = x + tn_1
        # t = x / (len(fp_test) * len(fp_train))
#训练集测试集相似度是0.269


    def __getitem__( self, index ):
        compounds_1 = self.simse[self.drug_1_smiles[index]]
        compounds_2 = self.simse[self.drug_2_smiles[index]]
        # compounds_1 = self.chem[self.drug_1_smiles[index]]
        # compounds_2 = self.chem[self.drug_2_smiles[index]]
        synergyScore = self.Y[index]#这里的affinitya(label)还是结合的分数吗？这里使用的label是二分类，如果是scores在模型中加一个sigmoid。
        fp1 = self.drug2fps[str(self.drug_1_smiles[index])]
        fp2 = self.drug2fps[str(self.drug_2_smiles[index])]
        context_features=self.features[self.context[index]]#从context中获取特征
        #context_feature= 输出的即为json文件里的一串list,细胞系留到以后处理，之后改变路径
        return [
                torch.FloatTensor(compounds_1),
                torch.FloatTensor(compounds_2),
                torch.LongTensor([synergyScore]),
                torch.FloatTensor([context_features]),
                torch.FloatTensor(fp1),
                torch.FloatTensor(fp2)]

