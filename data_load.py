'''
Created on 2018年11月26日

@author: coderwangson
'''
"#codeing=utf-8"

import numpy as np
import matplotlib.pyplot
import pandas as pd
def load_data():
    data = pd.read_csv("./data/mnist_train.csv",header=None)
    train_x = data.values[:,1:] 
    trian_y = data.values[:,0:1]
    data = pd.read_csv("./data/mnist_test.csv",header=None)
    test_x = data.values[:,1:] 
    test_y = data.values[:,0:1]
    return train_x.T,trian_y.T,test_x.T,test_y.T
