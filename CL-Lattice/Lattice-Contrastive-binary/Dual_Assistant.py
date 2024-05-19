import copy

import numpy as np
import sklearn.utils
from xgboost import XGBClassifier
from imblearn import over_sampling,under_sampling
def train_classifier(cmp_x, cmp_y):
    over = under_sampling.OneSidedSelection()
    cmp_x_resample, cmp_y_resample = over.fit_resample(cmp_x, cmp_y)
    cmp_x_resample, cmp_y_resample = sklearn.utils.shuffle(cmp_x_resample, cmp_y_resample)
    classifier = XGBClassifier().fit(cmp_x_resample, cmp_y_resample)
    return classifier
def Contrastive_Learning(history,new_configuration,pareto_frontier,hls):
    d = len(history[0].configuration)
    sampled_points = sampling(history, pareto_frontier, new_configuration)
    # sampled_points = copy.deepcopy(history)
    x_train,y_train = np.array([]).reshape(0,d),np.array([]).reshape(0,2)
    for design in sampled_points:
        x_train = np.vstack((x_train,design.configuration))
        y_train = np.vstack((y_train,[design.area,design.latency]))

    contrastive_x_train_coarse,contrastive_y_train_coarse,contrastive_x_train_fine,contrastive_y_train_fine = Enhance_Train(x_train,y_train,d)
    classifier_coarse = train_classifier(contrastive_x_train_coarse,contrastive_y_train_coarse)
    classifier_fine = train_classifier(contrastive_x_train_fine, contrastive_y_train_fine)

    contrastive_x,configurations = Enhance_Test(new_configuration,pareto_frontier,d)

    predict_dominances = classifier_coarse.predict(contrastive_x)

    coarse_dominance_graph,indexes_graph = construct(predict_dominances,len(pareto_frontier),len(configurations))

    coarse_dominance_graph_copy = copy.deepcopy(coarse_dominance_graph)
    fined_dominance_graph = refine(coarse_dominance_graph,contrastive_x,classifier_fine,indexes_graph,len(pareto_frontier),len(configurations))

    is_promising = identify(pareto_frontier,configurations,fined_dominance_graph)

    # test(fined_dominance_graph,hls,configurations,len(pareto_frontier),len(configurations))
    return is_promising
def sampling(history,pareto_frontier,new_configuration):
    threshold = 50
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
    K = 2
    N,NP = len(configurations),len(pareto_frontier)
    is_promising = []
    temp = []
    for i in range(NP,N):
        flag = True
        for j in range(N):
            if i == j:
                continue
            temp.append(graph_fixed[i,j])
            if graph_fixed[i,j] == 0:
                flag = False
        # if flag:
        if sum(temp) >= len(pareto_frontier)-K:
            is_promising.append(1)
        else:
            is_promising.append(0)
        temp = []
    return is_promising


def refine(coarse_dominance_graph,contrastive_x,classifier_fine,indexes_graph,NP,N):
    for i in range(N):
        for j in range(i+1,N):
            if (i < NP and j < NP):
                continue
            flag = False
            if coarse_dominance_graph[i,j] == 0 and coarse_dominance_graph[j,i] == 0:
                flag = True
            elif coarse_dominance_graph[i,j] == 1 and coarse_dominance_graph[j,i] == 0:
                flag = True
            elif coarse_dominance_graph[i,j] == 0 and coarse_dominance_graph[j,i] == 1:
                flag = True
            if flag:
                pi = classifier_fine.predict_proba(np.array([contrastive_x[int(indexes_graph[i, j])]]))[0]
                pj = classifier_fine.predict_proba(np.array([contrastive_x[int(indexes_graph[j, i])]]))[0]
                label_i = classifier_fine.predict(np.array([contrastive_x[int(indexes_graph[i, j])]]))[0]
                label_j = classifier_fine.predict(np.array([contrastive_x[int(indexes_graph[j, i])]]))[0]
                coarse_dominance_graph[i,j],coarse_dominance_graph[j,i] = label_i,label_j
                # if label_i == label_j:
                #     if pi[label_i] <= pj[label_j]:
                #         coarse_dominance_graph[i,j] = int(not(label_i))
                #     else:
                #         coarse_dominance_graph[j,i] = int(not(label_j))
    return coarse_dominance_graph


def test(graph,hls,configurations,NP,N):
    index = 0
    for i in range(N):
        for j in range(N):
            if (i < NP and j < NP) or (i == j):
                continue
            Li,Ai = hls.synthesise_configuration(configurations[i])
            Lj,Aj = hls.synthesise_configuration(configurations[j])
            # if ((Li <= Lj) and (Ai <= Aj)) and ((Li!=Lj) or (Ai!=Aj)):
            #     real = 1
            # elif ((Li >= Lj) and (Ai >= Aj)) and ((Li!=Lj) or (Ai!=Aj)):
            #     real = 0
            # else:
            #     real = 2

            if ((Li >= Lj) and (Ai >= Aj)):
                real = 0
            else:
                real = 1
            if 1:
                if i < NP:
                    print("P%d"%(i),end=" ")
                else:
                    print("C%d"%(i),end=" ")
                if j < NP:
                    print("P%d"%(j),end=" ")
                else:
                    print("C%d"%(j),end=" ")
                print("")
                print("real; predict;",(real,graph[i,j]))
                print("configurations:",configurations[i],configurations[j])
            index += 1
    print("")
    return graph


def construct(predict_dominances,NP,N):
    index = 0
    coarse_dominance_graph = np.empty((N,N),dtype=int)
    indexes_graph = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            if (i < NP and j < NP) or (i == j):
                indexes_graph[i,j] = -1
                continue
            coarse_dominance_graph[i,j] = predict_dominances[index]
            indexes_graph[i,j] = index
            index += 1
    return coarse_dominance_graph,indexes_graph

def Enhance_Train(x_train,y_train,d):
    contrastive_x_train_coarse,contrastive_y_train_coarse = np.array([]).reshape(0,2*d),np.array([]).reshape(0,1)
    contrastive_x_train_fine, contrastive_y_train_fine = np.array([]).reshape(0, 2 * d), np.array([]).reshape(0, 1)
    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[0]):
            if i == j:
                continue

            if (y_train[i,:] >= y_train[j,:]).all() or (y_train[i,:] <= y_train[j,:]).all():
                dominance = 0
            else:
                dominance = 1
            contrastive_x_train_coarse = np.vstack((contrastive_x_train_coarse, np.hstack((x_train[i, :], x_train[j, :]))))
            contrastive_y_train_coarse = np.vstack((contrastive_y_train_coarse,dominance))

            if (y_train[i,:] <= y_train[j,:]).all() and (y_train[i,:] != y_train[j,:]).any():
                dominance = 1
            elif (y_train[i,:] >= y_train[j,:]).all():
                dominance = 0
            else:
                continue
            contrastive_x_train_fine = np.vstack((contrastive_x_train_fine, np.hstack((x_train[i, :], x_train[j, :]))))
            contrastive_y_train_fine = np.vstack((contrastive_y_train_fine,dominance))

    return contrastive_x_train_coarse,contrastive_y_train_coarse,contrastive_x_train_fine,contrastive_y_train_fine
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
            if (i < len(pareto_frontier) and j < len(pareto_frontier)) or (i== j):
                continue
            x = np.hstack((configurations[i],configurations[j]))
            contrastive_x = np.vstack((contrastive_x,x))
    return contrastive_x,configurations

            



