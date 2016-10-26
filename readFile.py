import os, json
import pandas as pd
import csv
class readFile:
    
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def splitTarget(self):
        self.target = 'score_drug_gene_rank'
        self.gene_target = 'Target gene'
        self.rawData = pd.read_csv('featurized_data.csv')
        rawData = self.rawData.drop('Unnamed: 0', 1)
        rawData.shape
        outputs = self.rawData[self.target].values.tolist()
        inputs = self.rawData.drop([self.target, self.gene_target], 1).values.tolist()
        gene_labels = self.rawData[self.gene_target].values.tolist()
        return inputs, outputs, gene_labels

    #seperate function to rule out the reading of the data could have gone wrong
    def splitTarget2(self):
        self.target = 'score_drug_gene_rank'
        self.rawData = 'featurized_data_1.csv'
        count = 0
        IDs = []
        jsonfile = os.path.abspath(self.rawData)
        with open(jsonfile, 'rb') as data_file:    
            data = csv.DictReader(data_file)

            for row in data:
                inputrow = []
                outputrow = []
                for key, value in row.iteritems():
                    if str(key) == self.target:
                        outputrow = value
                    elif str(key) == 'Target gene':
                        IDs.append(value)
                    else:
                        inputrow.append(value)      
                self.inputs.append(inputrow)
                self.outputs.append(outputrow)
                count +=1
        return self.inputs, self.outputs, IDs