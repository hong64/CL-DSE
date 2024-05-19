import numpy as np
from Surrogate import Surrogate
from collections import Counter
from itertools import  product
import lattice_utils
from lattice_ds_point import DSpoint
from Test import  test
from ReadDB import readDB
X,Y,featureSets = readDB(7)
TP,FP,FN,TN,sum = 0,0,0,0,0
gbr0 = None
gbr1 = None
def DoubleGBR(n_estimators,synthesised_history, new_configuration, pareto_frontier,train,final = None):
    global gbr0,gbr1
    x = []
    y_area = []
    y_latency = []
    for ds_point in synthesised_history:
        x.append(ds_point.configuration)
        y_area.append(ds_point.area)
        y_latency.append(ds_point.latency)
    x,y_area, y_latency= np.array(x),np.array(y_area),np.array(y_latency)
    if final:
        if (not gbr0) or (not gbr1):
            gbr_area = Surrogate(x, y_area,n_estimators)
            gbr_latency = Surrogate(x, y_latency,n_estimators)
            gbr0 = gbr_area
            gbr1 = gbr_latency
        else:
            gbr_area = gbr0
            gbr_latency = gbr1
    elif not train:
        gbr_area = gbr0
        gbr_latency = gbr1
    else:
        gbr_area = Surrogate(x,y_area,n_estimators)
        gbr_latency = Surrogate(x,y_latency,n_estimators)
        gbr0 = gbr_area
        gbr1 = gbr_latency

    predict_configurations = _append(pareto_frontier,new_configuration)
    predict_area = gbr_area._predict(predict_configurations)
    predict_latency = gbr_latency._predict(predict_configurations)
    dominates = _relationship(predict_area,predict_latency)
    # assist(dominates,pareto_frontier,new_configuration)
    if final:
        return dominates,predict_latency[-1],predict_area[-1]
    else:
        return dominates
def assist(dominance,pareto_frontier,new_configuration):
    global TP,FP,FN,TN,sum
    sum += 1
    real_dominance = True
    id = np.where((X == new_configuration).all(1))[0][0]
    charact = Y[id,:]
    for point in pareto_frontier:
        if (charact > [point.latency,point.area]).all():
            real_dominance = False
            break

    if real_dominance and dominance:
        TP += 1
    elif dominance and not real_dominance:
        FP += 1
    elif real_dominance and not dominance:
        FN += 1
    elif not real_dominance and not dominance:
        TN += 1
def getMetrics():
    return TP,FP,FN,TN

def _append(pareto_frontier,new_configuration):
    configurations = [pareto_point.configuration for pareto_point in pareto_frontier]
    if np.array(new_configuration).ndim == 1:
        configurations.append(new_configuration)
    else:
        configurations.extend(new_configuration)
    return configurations
def _relationship(predict_area,predict_latency):
    diff = np.empty((predict_latency.shape[0],2))
    diff[:,0] = predict_area
    diff[:,1] = predict_latency
    #the new configuration's latency and area
    p = diff[-1,:]
    dominates = []
    for i in range(diff.shape[0]-1):
        if ((diff[i,:] - p) <= [0,0]).all() and ((diff[i,:] - p) != [0,0]).all():
            dominates.append(0)
        else:
            dominates.append(1)
    count = Counter(dominates)
    if count[1] == int(diff.shape[0]-1):
        return True
    else:
        return False

def predict_remain_configurations(n_estimators,discretized_descriptor,synthesised_history,pareto_frontier):
    global gbr0,gbr1
    synthesised_configurations = [ds_point.configuration for ds_point in synthesised_history]
    entire_configurations = list(product(*discretized_descriptor))
    entire_configurations = [list(config) for config in entire_configurations]
    synthesised_configurations = np.array(synthesised_configurations)
    entire_configurations = np.array(entire_configurations)
    remain_configurations = calArray2dDiff(entire_configurations,synthesised_configurations).tolist()
    dominated_remain_configurations = []

    gbr0, gbr1 = None, None
    for remain_configuration in remain_configurations:
        domindates,predict_latency,predict_area = DoubleGBR(n_estimators,synthesised_history, remain_configuration, pareto_frontier, False, True)
        if domindates:
            dominated_remain_configurations.append([remain_configuration,predict_latency,predict_area])

    non_dominated_offsprings = []
    for remain_point in dominated_remain_configurations:
        configuration,predict_latency,predict_area = remain_point[0],remain_point[1],remain_point[2]
        non_dominated_offsprings.append(DSpoint(predict_latency,predict_area,configuration))
    if len(non_dominated_offsprings):
        non_dominated_offsprings, pareto_frontier_idx = lattice_utils.pareto_frontier2d(non_dominated_offsprings)
    final_configuration = [p.configuration for p in non_dominated_offsprings]
    gbr0, gbr1 = None, None
    return final_configuration



def calArray2dDiff(array_0, array_1):
    array_0_rows = array_0.view([('', array_0.dtype)] * array_0.shape[1])
    array_1_rows = array_1.view([('', array_1.dtype)] * array_1.shape[1])

    return np.setdiff1d(array_0_rows, array_1_rows).view(array_0.dtype).reshape(-1, array_0.shape[1])

# def Select(synthesised_history):
#     pareto_frontier, pareto_frontier_idx = lattice_utils.pareto_frontier2d(synthesised_history)
#     domindated_points_idx = []
#     for i in range(len(synthesised_history)):
#         if i not in pareto_frontier_idx:
#             domindated_points_idx.append(i)
#     distances = np.empty((len(domindated_points_idx),len(pareto_frontier_idx)))
#     for i,dominated_id in enumerate(domindated_points_idx):
#         for j,pareto_id in enumerate(pareto_frontier_idx):
#             dist = np.array(synthesised_history[dominated_id].configuration) - np.array(synthesised_history[pareto_id].configuration)
#             distances[i,j] = np.linalg.norm(dist)
#     min_distances = np.min(distances,axis=1)
#     idx = np.argsort(min_distances,axis=0)
#     n = min(50,idx.shape[0])
#     for i in range(n):
#         pareto_frontier.append(synthesised_history[domindated_points_idx[idx[i]]])
#     return pareto_frontier



















