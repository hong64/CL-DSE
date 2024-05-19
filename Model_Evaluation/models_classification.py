import copy

import  numpy as np
def construct_graph(N,probs,labels):
    cnt = 0
    indexes = np.empty((N,N))
    graph = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            graph[i,j] = labels[cnt]
            indexes[i,j] = cnt
            cnt += 1

    #Fix the conflicts in the graph
    for i in range(N):
        for j in range(i + 1, N):
            if graph[i, j] == 0 and graph[j, i] == 0:
                probi = probs[int(indexes[i, j])][int(graph[i, j])]
                probj = probs[int(indexes[j, i])][int(graph[j, i])]
                if probi < probj:
                    graph[i, j] = 1
                else:
                    graph[j, i] = 1
    return graph

def test(graph, contrastive_y_test):
    cnt = 0
    pos = 0
    n = graph.shape[0]
    real_graph = np.empty((n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            real_graph[i,j] = contrastive_y_test[idx]
            idx+=1
    for i in range(n):
        for j in range(i+1,n):
            if (real_graph[i,j] == 1 and real_graph[j,i] == 1) or (real_graph[i,j] == 1 and real_graph[j,i] == 0):
                real_dominance = 1
            else:
                real_dominance = 0

            if (graph[i,j] == 1 and graph[j,i] == 1) or (graph[i,j] == 1 and graph[j,i] == 0):
                predict_dominance = 1
            else:
                predict_dominance = 0

            if real_dominance == predict_dominance:
                pos+=1
            cnt+=1
    return pos/cnt

def Contrastive_Pairs(x, y, d):
    contrastive_x, contrastive_y = np.array([]).reshape(0, 2 * d), np.array([]).reshape(0, 1)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if i == j:
                continue

            contrastive_x = np.vstack(
                (contrastive_x, np.hstack((x[i, :], x[i, :] - x[j, :]))))
            if (y[i, :] >= y[j, :]).all():
                dominance = 0
            else:
                dominance = 1

            contrastive_y = np.vstack((contrastive_y, dominance))
    return contrastive_x, contrastive_y






def Contrastive_Models(samples,selected_models):
    predict_results = {name:{} for name in selected_models.keys()}
    contrastive_x_train, contrastive_y_train = Contrastive_Pairs(samples.x_train, samples.y_train, samples.d)
    contrastive_x_test, contrastive_y_test = Contrastive_Pairs(samples.x_test,samples.y_test,samples.d)
    model_accs = dict.fromkeys(selected_models.keys(),-1)
    for name,model in selected_models.items():
        mc = copy.deepcopy(model)
        mc.random_state = 1
        m = mc.fit(contrastive_x_train,contrastive_y_train)
        probs = m.predict_proba(contrastive_x_test)
        labels = m.predict(contrastive_x_test)
        graph = construct_graph(samples.x_test.shape[0],probs,labels)
        acc = test(graph, contrastive_y_test)
        model_accs[name] = acc
        predict_results[name]["probs"] = probs
        predict_results[name]["labels"] = labels
        predict_results[name]["enhance"] = {
            "contrastive_x_train":contrastive_x_train,
            "contrastive_y_train":contrastive_y_train,
            "contrastive_x_test":contrastive_x_test,
            "contrastive_y_test":contrastive_y_test
        }
        predict_results[name]["acc"] = acc
        predict_results[name]["graph"] = graph
    return model_accs,predict_results