import csv
from sklearn import preprocessing
import numpy as np
THRESHOLD = 0.72
import cv2
import os


class Dataset2:
    def __init__(self):
        pass




    def fetch_data_and_proc(self,LIMCOUNT):
        from keras.applications import vgg16
        from keras import backend as K

        mod = vgg16.VGG16(include_top=False,weights="imagenet")
        count = 0
        path = "../../datasets/raw_data/fruits-360/Training/"
        X = []
        Y = []
        for fold in os.listdir(path):
            #print fold
            lim = 0
            for fin in os.listdir(path+fold):
                    
                    im = cv2.imread(path+fold+"/"+fin)
                    im = cv2.resize(im,(224,224))
                    #cv2.imshow('pic',im)
                    #cv2.waitKey(10)
                    dat = []
                    dat.append(im)
                    dat = np.array(dat)
                    #print dat.shape
                    
                    out = mod.predict(dat)
                    #print out.flatten().shape
                    #cv2.imshow('pic2',out[0][:][:][:2])
                    #cv2.waitKey(1000)
                    X.append(out.flatten())
                    Y.append(count)
                    if lim > LIMCOUNT:
                        break
                    lim += 1
            count += 1
            print fold
                    
                    

        X = np.array(X)
        Y = np.array(Y)
    
        print "done loading data"
        del mod
        K.clear_session()
        if (LIMCOUNT == 0):
            X = np.load('../data.npy')
            Y = np.load('../labels.npy')
        
        else:
            np.save("../data"+str(LIMCOUNT)+".npy",X)
            np.save("../labels"+str(LIMCOUNT)+".npy",Y)
        return X,Y

    def fetch_data(self,numItems):
        if numItems == 0:
            X = np.load('../data.npy')
            Y = np.load('../labels.npy')
            return X,Y

        fnameD = "../data" + str(numItems) + ".npy"
        fnameL = "../labels" + str(numItems) + ".npy"
        X = np.load(fnameD)
        Y = np.load(fnameL)
        return X,Y
