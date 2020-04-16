#!/usr/bin/env python3

import os
import sys
import glob
import time
import multiprocessing as mp
import pandas as pd
import numpy as np


#from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn import tree

pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=3)

class Config:
    def __init__(self, input_csv, cluster_csv, label_csv, train_size, columns, dists):
        self.input_csv = input_csv
        self.cluster_csv = cluster_csv 
        self.label_csv = label_csv 
        self.columns = columns, 
        self.train_size = train_size,
        self.dists = dists



def predict(data, dist, cfg):
    cols = cfg.columns[0]
    predict_data = data[cols]
    predict_data = predict_data[0:cfg.train_size[0]]
    X = Normalizer().fit_transform(predict_data)
    model= AgglomerativeClustering(n_clusters=None, distance_threshold=dist)
    model.fit(X)
    y_pred = model.labels_.astype(np.int)
    predict_data['cluster'] = y_pred
    predict_data['dist'] = dist
    #predict_data.reset_index(inplace=True)
    #predict_data.set_index(['jobid', 'dist'], inplace=True)
    return predict_data


if __name__ == '__main__':
    CONFIG = Config(
        input_csv='../../datasets/job-metrics.csv',
        cluster_csv='../../datasets/job-metrics_clustered.csv',
        label_csv='../../datasets/job-metrics_labeled.csv',
        train_size = 10000,
        columns = ['utilization', 'problem_time', 'balance_x', 'elapsed', 'ntasks', 'total_nodes'],
        dists=list(np.arange(0.1, 1, 0.1))
        )

    DATA = pd.read_csv(CONFIG.input_csv, index_col='jobid', usecols=['jobid', 'utilization', 'problem_time', 'balance_x', 'elapsed', 'ntasks', 'total_nodes', 'job_name'], nrows=1000000)
    DATA = DATA[DATA['utilization'] != 0]
    DATA['ntasks'] = (DATA['ntasks'] / DATA['total_nodes']).astype(np.int64)
    DATA.dropna(inplace=True)
    RES = list()
    for dist in CONFIG.dists:
        start = time.time()
        RES.append(predict(DATA, dist, CONFIG))
        stop = time.time()
        print('Duration %f seconds' % (stop - start))
    RES_DF = pd.concat(RES)
    RES_DF.to_csv(CONFIG.cluster_csv)


    RES_DF = RES_DF[RES_DF['dist'] == 0.5]
    X = RES_DF[CONFIG.columns[0]]
    Y = RES_DF['cluster']
    CLF = tree.DecisionTreeClassifier()
    CLF = CLF.fit(X, Y)

    LABELS = CLF.predict(DATA[CONFIG.columns[0]])
    DATA['labels'] = LABELS
    DATA.to_csv(CONFIG.label_csv)
