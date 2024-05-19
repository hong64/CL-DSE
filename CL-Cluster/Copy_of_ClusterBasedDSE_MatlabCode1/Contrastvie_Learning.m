import numpy as np
from xgboost import XGBClassifier
from imblearn import over_sampling
def Contrastive_Learning(history,new_configuration,pareto_frontier):
    n,d = len(history),len(history[0].configuration)
    x_train,y_train = np.array([]).reshape(0,d),np.array([]).reshape(0,2)
    for design in history:
        x_train = np.vstack((x_train,design.configuration))
        y_train = np.vstack((y_train,[design.area,design.latency]))

    contrastive_x_train,contrastive_y_train = Enhance_Train(x_train,y_train,d)
    over = over_sampling.RandomOverSampler(random_state=1)
    contrastive_x_train_resample, contrastive_y_train_resample = over.fit_resample(contrastive_x_train,contrastive_y_train)
    classifier = XGBClassifier().fit(contrastive_x_train_resample,contrastive_y_train_resample)

    contrastive_x,configurations = Enhance_Test(new_configuration,pareto_frontier,d)

    predict_dominances = classifier.predict(contrastive_x)

    graph = construct(predict_dominances,len(pareto_frontier),len(configurations))

    graph_fixed = fix(graph,pareto_frontier,configurations)

    is_promising = identify(pareto_frontier,configurations,graph_fixed)
    return is_promising
def identify(pareto_frontier,configurations,graph_fixed):
    N,NP = len(configurations),len(pareto_frontier)
    is_promising = []
    for i in range(N):
        if i < NP:
            continue
        if (graph_fixed[i,:] == 1).all():
            is_promising.append(1)
        else:
            is_promising.append(0)
    return is_promising


def fix(graph,pareto_frontier,configurations):
    N = len(configurations)
    pareto_configurations = np.array([p.configuration for p in pareto_frontier])
    for i in range(N):
        for j in range(i+1,N):
            if (i<len(pareto_frontier) and (j<len(pareto_frontier))):
                continue
            if graph[i,j] == 0 and graph[j,i] == 0:
                ci,cj = np.array([configurations[i]]),np.array([configurations[j]])
                di = np.reshape(np.sum(ci ** 2, axis=1), (ci.shape[0], 1)) + np.sum(pareto_configurations ** 2,
                                                                axis=1) - 2 * ci.dot(pareto_configurations.T)
                dj = np.reshape(np.sum(cj ** 2, axis=1), (cj.shape[0], 1)) + np.sum(pareto_configurations ** 2,
                                                                axis=1) - 2 * cj.dot(pareto_configurations.T)
                mi,mj = np.min(di),np.min(dj)
                if mi < mj:
                    graph[i,j],graph[j,i]= 1,0
                elif mj < mi:
                    graph[i,j],graph[j,i] = 0,1
                else:
                    graph[i,j],graph[j,i] = 1,1
    return graph




def construct(predict_dominances,NP,N):
    index = 0
    graph = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            if (i < NP and j < NP) or (i == j):
                continue
            graph[i,j] = predict_dominances[index]
            index += 1
    return graph

def Enhance_Train(x_train,y_train,d):
    contrastive_x_train,contrastive_y_train = np.array([]).reshape(0,2*d),np.array([]).reshape(0,1)
    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[0]):
            if i == j:
                continue
            contrastive_x_train = np.vstack(( contrastive_x_train, np.hstack((x_train[i,:],x_train[j,:])) ))
            if (y_train[i,:] >= y_train[j,:]).all() and (y_train[i,:] != y_train[j,:]).any():
                dominance = 0
            else:
                dominance = 1
            contrastive_y_train = np.vstack((contrastive_y_train,dominance))
    return contrastive_x_train,contrastive_y_train
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

            



