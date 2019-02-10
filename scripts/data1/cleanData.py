import csv
from sklearn import preprocessing
import numpy as np
THRESHOLD = 0.72



class Dataset1:
    def __init__(self):
        pass



    def fetch_data_bin(self):
        X = []
        Y = []
        with open('../../datasets/raw_data/data1.csv') as csvfile:
            datareader = csv.reader(csvfile)

            for row in datareader:
                #print row
                data = row[1:]
                if float(data[-1]) >= THRESHOLD:
                    data[-1] = 1
                else:
                    data[-1] = 0

                
                feats = data[:-1]
                label = data[-1]
                #print [float(i) for i in data]
                X.append([float(i) for i in feats])
                Y.append([float(label)])

            X = np.array(X)
            Y = np.array(Y)
        

            return X,Y

    def fetch_data_multi(self):
        X = []
        Y = []
        with open('../../datasets/raw_data/data1.csv') as csvfile:
            datareader = csv.reader(csvfile)

            for row in datareader:
                #print row
                data = row[1:]
                if float(data[-1]) <= 0.7:
                    data[-1] = 0
                elif float(data[-1]) <= 0.85:
                    data[-1] = 1
                else:
                    data[-1] = 2

            
                feats = data[:-1]
                label = data[-1]
                #print [float(i) for i in data]
                X.append([float(i) for i in feats])
                Y.append([float(label)])

        X = np.array(X)
        Y = np.array(Y)
    

        return X,Y