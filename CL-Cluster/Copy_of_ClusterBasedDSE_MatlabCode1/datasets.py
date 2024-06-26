import pymysql
import pandas as pd
import re
import sqlite3
import numpy as np
from Encode import  encode
def _read(name):
    conn = sqlite3.connect("./dbs/"+name+'.db')
    sql = "select * from "+name
    data = pd.read_sql(sql, conn)
    return data
def Area(data):
    ff_a = 548160
    lut_a = 274080
    dsp_a = 2520
    bram_a = 1824
    ff = data[["hls_ff"]]/ff_a
    lut = data[["hls_lut"]]/lut_a
    bram = data[["hls_bram"]]/bram_a
    dsp  = data[["hls_dsp"]]/dsp_a
    return np.array((ff["hls_ff"] + lut["hls_lut"] + dsp["hls_dsp"] + bram["hls_bram"])/4)

def preprocess(data):
    area = Area(data)
    area = np.expand_dims(area, axis=1)
    # area = np.expand_dims(data[["hls_ff", "hls_lut", "hls_bram", "hls_dsp"]].mean(axis=1).values / 4, axis=1)
    # area = np.expand_dims(data[["hls_ff"]].mean(axis=1).values,axis=1)
    latency = np.array(data[["average_latency"]])
    # y = np.array(data[["hls_ff","average_latency"]])
    y = np.hstack((area, latency))
    selected_columns = []
    columns = data.columns.tolist()
    for column in columns:
        if column[0] == "f":
            selected_columns.append(column)
    x_ = np.array(data[selected_columns])
    x,discretized_feature,original_feature= encode(x_)
    status = np.array(data["hls_exit_value"])
    cluster_data, featureSets, discretizedFeatureSets = Cluster(x,y,discretized_feature,original_feature)
    return cluster_data,status.tolist(),featureSets,discretizedFeatureSets
def get_dse_description(name):
    data = _read(name)
    cluster_data, status, featureSets, discretizedFeatureSets = preprocess(data)
    # x,y,status,discretized_feature,original_feature = preprocess(data)
    return cluster_data, status, featureSets, discretizedFeatureSets
def Cluster(x,y,discretized_feature,original_feature):
    data = np.hstack((np.expand_dims(y[:,0],axis=1),x))
    data = np.hstack((np.expand_dims(y[:,1],axis=1),data))
    discretizedFeatureSets = [discretized_feature[i].tolist() for i in range(discretized_feature.shape[0])]
    return data.tolist(),original_feature,discretizedFeatureSets

