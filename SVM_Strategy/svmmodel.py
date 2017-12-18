from sklearn import svm
from sklearn.externals import joblib
# from sklearn.ensemble import 
from helperFunctions import read_h5, addLabel, compute_average
import talib as tb

import h5py
import tables
import pandas as pd
import numpy as np

kd_low = 10
kd_high = 90
fastk_period = 9
slowk_period = 3
slowd_period = 3

path_data= '/Users/lucas/Desktop/workspace/PythonPlatform/Data/data_format1_20170717_20170915.h5'
path_info= '/Users/lucas/Desktop/workspace/PythonPlatform/Data/information.csv'

indexes = ['A.DCE', 'AG.SHF', 'AU.SHF', 'I.DCE', 'IC.CFE', 'IF.CFE',
           'IH.CFE', 'J.DCE', 'JM.DCE', 'M.DCE', 'RB.SHF', 'Y.DCE', 'ZC.CZC']

target_index = [12, 10, 7, 3]

dict = read_h5(path_data)
#train_label = np.zeros(shape=(4, 8775))

flagBuy = 0
flagSell = 0

for i in target_index:
    keyData = dict[indexes[i]].values[:, 0:4]
    #print keyData

    train_label = list()
    train_label.append(0)


    # For the average
    average_label = list()
    average_label.append(0)
    average_target = compute_average(keyData).reshape(-1, 1)
    average_label += addLabel(average_target)


    # For the close
    close_label = list()
    close_label.append(0)
    close_target = keyData[:, 3].reshape(-1, 1)
    #print keyData[:, 3].reshape(-1, 1)
    close_label += addLabel(close_target)


    # For the open
    open_label = list()
    open_label.append(0)
    open_target = keyData[:, 0].reshape(-1, 1)
    open_label += addLabel(open_target)

    # Train label
    train_label = list()

    for j in range(len(average_label)):
        if average_label[j] == close_label[j] == open_label[j]:
           train_label.append(1)
        else:
           train_label.append(0)



    # C1 -- average, C2 -- close
    train_input = np.concatenate((average_target, close_target, open_target), axis=1).reshape(len(average_target), 3)
    train_label = np.asarray(train_label)

    # SVM
    clf = svm.SVC()
    clf.fit(train_input, train_label)

    joblib.dump(clf, 'svmindex' + str(i) + '.pkl')



    # print close_target.shape
    # print average_target.shape

    # train_input = np.array([len(average_target), 2])





    # print train_input.shape

    # print average_target

    # print close_target
    # print "ddddddddd"
    # print train_input
    # print "ooooooo"
    # print train_input.shape ## (x,)

    # train_label += (binarize(average_label == close_target))









