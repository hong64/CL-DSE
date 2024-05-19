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
    hls_time = np.array(data["hls_execution_time"])
    return x,y,status,discretized_feature,original_feature,hls_time
def get_dse_description(name):
    data = _read(name)
    x,y,status,discretized_feature,original_feature,hls_time = preprocess(data)
    return x,y,status,discretized_feature,original_feature,hls_time
def calculate_cluster_mean_time(data, name, n):
    mean_time = 0
    x, y, status, discretized_feature, original_feature, hls_time = get_dse_description(name)
    for j in range(n):
        time = 0
        sample_data = np.unique(data['sampledDataEvolution'].tolist()[0][j],axis=0)

        for i in range(len(sample_data)):
            if (sample_data[i, 0:2] == [10000000, 10000000]).all():
                continue
            elif (sample_data[i, 0:2] == [20000000, 20000000]).all():
                time += 0
            else:
                id = np.where((x == sample_data[i, 2:]).all(1))[0].tolist()[0]
                time += hls_time[id]
        mean_time+=time
    return mean_time/n

def calculate_lattice_mean_time(data, name, n):
    mean_time = 0
    x, y, status, discretized_feature, original_feature, hls_time = get_dse_description(name)
    for j in range(n):
        time = 0
        for item in data[j]:
            id = list(item.keys())[0]
            if not item[id]:
                continue
            time+=hls_time[id]
        mean_time+=time
    return mean_time/n

def calculate_EGA_mean_time(data, name, n):
    mean_time = 0
    x, y, status, discretized_feature, original_feature, hls_time = get_dse_description(name)
    for i in range(n):
        time = 0
        visited_design_ids = data[i]
        for id in visited_design_ids:
            if (y[id,:]!=[10000000,10000000]).all():
                time += hls_time[id]
            else:
                time += 0
        mean_time += time
    return mean_time/n
from lattice_ds_point import DSpoint
import lattice_utils
def obtain_pareto_exhaustive_frontier(x,y):
    synthesis_result = y.tolist()
    configurations = x.tolist()

    # Format data for the exploration
    entire_ds = []
    for i in range(0, len(synthesis_result)):
        entire_ds.append(DSpoint(synthesis_result[i][1], synthesis_result[i][0], list(configurations[i])))
    pareto_frontier_exhaustive, pareto_frontier_exhaustive_idx = lattice_utils.pareto_frontier2d(entire_ds)
    return pareto_frontier_exhaustive,pareto_frontier_exhaustive_idx
def compare_cluster_pareto_points(data, name, n):
    x, y, status, discretized_feature, original_feature, hls_time = get_dse_description(name)
    pareto_frontier_exhaustive,pareto_frontier_exhaustive_idx = obtain_pareto_exhaustive_frontier(x,y)
    average_num_of_paretos = 0
    for j in range(n):
        num = 0
        sample_data = np.unique(data['sampledDataEvolution'].tolist()[0][j], axis=0)
        for i in range(len(sample_data)):
            id = np.where((x==sample_data[i,2:]).all(1))[0].tolist()[0]
            if id in pareto_frontier_exhaustive_idx:
                num+=1
        average_num_of_paretos += num
    return average_num_of_paretos/n

def compare_lattice_pareto_points(data, name, n):
    x, y, status, discretized_feature, original_feature, hls_time = get_dse_description(name)
    pareto_frontier_exhaustive,pareto_frontier_exhaustive_idx = obtain_pareto_exhaustive_frontier(x,y)
    avergae_num_of_paretos = 0
    for j in range(n):
        num = 0
        for item in data[j]:
            id = list(item.keys())[0]
            if id in pareto_frontier_exhaustive_idx:
                num+=1
        avergae_num_of_paretos+=num
    return avergae_num_of_paretos/n
def compare_EGA_pareto_points(data,name,n):
    average_num_of_paretos = 0
    x, y, status, discretized_feature, original_feature, hls_time = get_dse_description(name)
    pareto_frontier_exhaustive,pareto_frontier_exhaustive_idx = obtain_pareto_exhaustive_frontier(x, y)
    for i in range(n):
        num = 0
        visited_design_ids = data[i]
        for id in visited_design_ids:
            if id in pareto_frontier_exhaustive_idx:
                num+=1
        average_num_of_paretos += num
    return average_num_of_paretos / n






