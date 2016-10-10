import os, json
import pandas as pd
class readFile:
    
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def splitTarget(self):
        self.target = 'score_drug_gene_rank'
        self.gene_target = 'Target gene'
        self.rawData = pd.read_csv('featurized_data.csv')
        rawData = self.rawData.drop('Unnamed: 0', 1)
        outputs = self.rawData[self.target].values.tolist()
        inputs = self.rawData.drop([self.target, self.gene_target], 1).values.tolist()
        gene_labels = self.rawData[self.gene_target].values.tolist()
        return inputs, outputs, gene_labels
        