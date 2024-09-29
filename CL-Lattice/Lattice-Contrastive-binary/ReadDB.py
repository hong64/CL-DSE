#导入sqllite3模块
import sqlite3
import pandas as pd
import numpy as np
import copy
path = "./datasets/"
def readDB(benchmarkID):
    global path
    names = ["ChenIDct", "adpcm_decode", "adpcm_encode", "Autocorrelation", "Reflection_coefficients"]
    names_append = ["aes256_encrypt_ecb","ellpack","matrix","merge","aes_notable"]
    names.extend(names_append)
    name = names[benchmarkID]
    # 1.硬盘上创建连接
    con = sqlite3.connect(path+name+'.db')
    # 获取cursor对象
    cur = con.cursor()
    # 执行sql读取数据
    sql = 'SELECT * FROM ' + name
    data = pd.read_sql_query(sql, con)
    # 删除无用列

    if name in ["adpcm_decode","adpcm_encode","Autocorrelation","ChenIDct","Reflection_coefficients"]:
        if name != "Reflection_coefficients":
            data = data.drop(['solutions', 'intervals', 'luts', 'bundle_a'], axis=1)
        else:
            data = data.drop(['solutions','luts','loop0','bundle_a'],axis = 1)
    if name in names_append:
        data = data.drop(['hls_lut','hls_dsp','hls_bram'],axis=1)

    #定义decision space
    featureSets = []
    #640
    if name == "adpcm_decode":
        featureSets = np.array([[0, 5, 10, 25, 50],[0, 2, 5, 10],[0, 2, 5, 10],[0, 1],[0, 6],[0, 1]],dtype=object)
    #640
    elif name == "adpcm_encode":
        featureSets = np.array([[0, 5, 10, 25, 50], [0, 2, 5, 10], [0, 2, 11, 22], [0, 1], [0, 6], [0, 1]], dtype=object)
    #1728
    elif name == "Autocorrelation":
        featureSets = np.array([[0, 4, 16, 40, 80, 160],[0, 4, 16, 40, 80, 160],[0, 9],[0, 4, 19, 38, 76, 152],[0, 9],[0, 1]], dtype=object)
    #224
    elif name == "ChenIDct":
        featureSets = np.array([[0, 2, 4, 8],[0, 2, 4, 8],[0, 2, 4, 8, 16, 32, 64],[0, 1]], dtype=object)
    #4608
    elif name == "Reflection_coefficients":
        featureSets = np.array([[0, 3, 9],[0, 7],[0, 3, 9],[0, 2, 4, 8],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]], dtype=object)
    #3072
    elif name == "aes256_encrypt_ecb":
        featureSets = np.array([[0, 1], [0, 1], [0, 1], [2, 4], [0, 1], [1, 2, 4, 8, 16, 32], [1, 2, 4, 8], [1, 13],
                                   [0, 1]],dtype=object)
    #1600
    elif name == "ellpack":
        featureSets = np.array([[1, 2, 4,10,20],[1,2,13,19,26],[1,2,13,19],[1,2,13,19],[1,2,5,10]],dtype=object)
    #1372
    elif name == "matrix":
        featureSets = np.array([[1,2,4,8,16,32,64],[1,2,4,8,16,32,64],[1,13],[1,2,4,8,16,32,64],[0,1]],dtype=object)
    #4096
    elif name == "merge":
        featureSets = np.array([[1,2,4,8,16,32,64,128],[1,2,4,8,16,32,64,128],[1,2,4,8,16,32,64,128],[1,2,4,8,16,32,64,128]],dtype=object)
    #12288
    elif name == "aes_notable":
        featureSets = np.array([[0, 1], [0, 1], [0, 1], [2, 4], [0, 1], [1, 2, 4, 8, 16, 32], [1, 2, 4, 8], [1, 13], [0,1], [0,1], [0,1]],dtype = object)
    x = data.iloc[:, 2:]
    y = data.iloc[:, 0:2]
    X = np.array(x)
    Y = np.array(y)
    if name in names_append:
        X = X.astype(int)
        Y = Y.astype(int)

    X,featureSets = convert(X,featureSets)
    if benchmarkID == 0:
        CX = copy.deepcopy(X)
        X[:, 0] = CX[:, 2]
        X[:, 1] = CX[:, 0]
        X[:, 2] = CX[:, 1]
    if benchmarkID == 3:
        CX = copy.deepcopy(X)
        X[:, 2] = CX[:, 3]
        X[:, 3] = CX[:, 2]
    return X,Y,featureSets
def convert(X,featureSets):
    discretized_feature = discretize_dataset(featureSets)
    d = np.empty((X.shape[0],X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            index = featureSets[j].index(X[i][j])
            d[i,j] = discretized_feature[j][index]
    X = d
    discretized_feature = np.array(discretized_feature,dtype=np.object)
    return X,discretized_feature
def discretize_dataset( lattice_descriptor):
    discretized_feature = []
    for feature in lattice_descriptor:
        tmp = []
        for x in frange(0, 1, 1. / (len(feature) - 1)):
            tmp.append(x)
        discretized_feature.append(tmp)

    return discretized_feature

def frange(start, stop, step):
    x = start
    output = []
    while x <= stop:
        output.append(x)
        x += step
    output[-1] = 1
    return output





