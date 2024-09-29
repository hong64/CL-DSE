import pandas as pd
import sqlite3
import numpy as np
from Encode import encode
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
    latency = np.array(data[["average_latency"]])
    y = np.hstack((area, latency))
    selected_columns = []
    columns = data.columns.tolist()
    for column in columns:
        if column[0] == "f":
            selected_columns.append(column)
    x_ = np.array(data[selected_columns])
    x,discretized_feature,original_feature= encode(x_)
    status = np.array(data["hls_exit_value"])
    return x_,y,status,discretized_feature,original_feature
def get_dse_description(name):
    data = _read(name)
    x,y,status,discretized_feature,original_feature = preprocess(data)
    return x,y,status,discretized_feature,original_feature

