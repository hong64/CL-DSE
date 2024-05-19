import copy
import random

import numpy as np
import sklearn.utils
from xgboost import XGBClassifier
train_interval = 1
class TrainCondition:
    def __init__(self, interval, classifier):
        self.interval = interval
        self.classifier = classifier
def Contrastive_Learning(history,new_configuration,pareto_frontier,train):
    d = len(history[0].configuration)
    if train.interval == train_interval:
        train.interval = 0
        sampled_points = sampling(history,pareto_frontier,new_configuration)
        x_train,y_train = np.array([]).reshape(0,d),np.array([]).reshape(0,2)
        for design in sampled_points:
            x_train = np.vstack((x_train,design.configuration))
            y_train = np.vstack((y_train,[design.area,design.latency]))

        contrastive_x_train,contrastive_y_train = Enhance_Train(x_train,y_train,d)
        contrastive_x_train, contrastive_y_train = sklearn.utils.shuffle(contrastive_x_train,contrastive_y_train,random_state=1)
        if (np.array([np.where(contrastive_y_train == 0)[0].shape[0], np.where(contrastive_y_train == 1)[0].shape[0]]) == [0, 0]).any():
            train.interval = train_interval
            train.classifier = None
            return [1],train
        classifier = XGBClassifier(n_estimators=200,random_state=1).fit(contrastive_x_train,contrastive_y_train)
        train.classifier = classifier

    classifier = train.classifier
    contrastive_x,configurations = Enhance_Test(new_configuration,pareto_frontier,d)
    predict_dominances_proba = classifier.predict_proba(contrastive_x)
    predict_dominances = classifier.predict(contrastive_x)
    graph,indexes_graph = construct(predict_dominances,len(pareto_frontier),len(configurations))
    graph_fixed = fix(graph,pareto_frontier,configurations,predict_dominances_proba,indexes_graph)
    is_promising = identify(pareto_frontier,configurations,graph_fixed)
    if is_promising[0] == 1:
        train.interval += 1
    return is_promising,train
def sampling(history,pareto_frontier,new_configuration):
    threshold = 500
    pareto_configurations = [p.configuration for p in pareto_frontier]
    history_configurations = np.array([h.configuration for h in history])
    sampled_points = copy.deepcopy(pareto_frontier)
    new_configuration = np.array([new_configuration])
    dists = np.reshape(np.sum(history_configurations ** 2, axis=1), (history_configurations.shape[0], 1)) + \
            np.sum(new_configuration ** 2,axis=1) - 2 * history_configurations.dot(new_configuration.T)
    min_dists = np.min(dists,axis=1)
    sorted_indexes = np.argsort(min_dists).tolist()
    if len(history) - len(pareto_frontier) <= threshold:
        return history
    count = 0
    for index in sorted_indexes:
        if history[index].configuration in pareto_configurations:
            continue
        sampled_points.append(history[index])
        count += 1
        if count == threshold:
            break
    return sampled_points

def identify(pareto_frontier,configurations,graph_fixed):
    N,NP = len(configurations),len(pareto_frontier)
    is_promising = []
    for i in range(NP,N):
        is_pareto = True
        for j in range(N):
            if i == j:
                continue
            if graph_fixed[i,j] == 0:
                is_pareto = False
                break
        if is_pareto:
            is_promising.append(1)
        else:
            is_promising.append(0)
    return is_promising
def fix(graph,pareto_frontier,configurations,predict_dominances_proba,indexes_graph):
    N = len(configurations)
    for i in range(N):
        for j in range(i+1,N):
            if (i<len(pareto_frontier) and (j<len(pareto_frontier))):
                continue
            if (graph[i,j] == 0 and graph[j,i] == 0):
                probi = predict_dominances_proba[int(indexes_graph[i, j])][int(graph[i, j])]
                probj = predict_dominances_proba[int(indexes_graph[j, i])][int(graph[j, i])]
                if probi < probj:
                    graph[i,j] = 1
                else:
                    graph[j,i] = 1

    return graph





def construct(predict_dominances,NP,N):
    index = 0
    graph = np.empty((N,N))
    indexes_graph = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            if (i < NP and j < NP) or (i == j):
                continue
            graph[i,j] = predict_dominances[index]
            indexes_graph[i,j] = index
            index += 1
    return graph,indexes_graph

def Enhance_Train(x_train,y_train,d):
    contrastive_x_train = np.array([]).reshape(0, 2 * d)
    contrastive_y_train = np.array([]).reshape(0, 1)
    for i in range(x_train.shape[0]):
        temp = np.ones((x_train.shape[0] - 1, 1))
        flag = [True] * x_train.shape[0]
        flag[i] = False
        diff_x = x_train[i, :] - x_train[flag, :]
        xi = np.tile(x_train[i, :], (diff_x.shape[0], 1))
        contrastive_x_train = np.vstack((contrastive_x_train, np.hstack((xi, diff_x))))
        diff_y = y_train[i, :] - y_train[flag, :]
        dominated_ids = np.where((diff_y >= [0, 0]).all(1))[0].tolist()
        temp[dominated_ids, :] = 0
        contrastive_y_train = np.vstack((contrastive_y_train, temp))

    return contrastive_x_train, contrastive_y_train
def Enhance_Test(new_configuration,pareto_frontier,d):
    contrastive_x = np.array([]).reshape(0,2*d)
    configurations = [p.configuration for p in pareto_frontier]
    if np.array(new_configuration).ndim == 1:
        configurations.append(new_configuration)
    else:
        for configuration in new_configuration:
            configurations.append(configuration)
    
    for i in range(len(configurations)):
        for j in range(len(configurations)):
            if (i < len(pareto_frontier) and j < len(pareto_frontier)) or (i == j):
                continue
            x = np.hstack((configurations[i],np.array(configurations[i])-np.array(configurations[j])))
            contrastive_x = np.vstack((contrastive_x,x))
    return contrastive_x,configurations

            



