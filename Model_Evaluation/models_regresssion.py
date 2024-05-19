import copy
import numpy as np
def cal_acc(areas,latencys,samples_y):
    cnt = 0
    acc = 0
    n = len(areas)
    for i in range(n):
        for j in range(i+1,n):
            if (samples_y[i,:] >= samples_y[j,:]).all():
                real = 0
            else:
                real = 1
            if (np.array([areas[i],latencys[i]]) >= np.array([areas[j],latencys[j]])).all():
                predict = 0
            else:
                predict = 1
            if real == predict:
                acc+=1
            cnt +=1
    return acc/cnt


def Regression_Models(selected_regression_models,samples_x,samples_y,train_index,test_index,regression_models):
    models_acc = dict.fromkeys(selected_regression_models.keys(),0)
    predict_results = {name:{} for name in selected_regression_models.keys()}
    for name in selected_regression_models.keys():
        area_model = copy.deepcopy(regression_models[name])
        latency_model = copy.deepcopy(regression_models[name])
        area_model.random_state=1
        latency_model.random_state=1
        predictions_area = area_model.fit(samples_x[train_index, :],samples_y[train_index,0]).predict(samples_x[test_index,:])
        predictions_latency = latency_model.fit(samples_x[train_index, :],samples_y[train_index,1]).predict(samples_x[test_index,:])
        areas = predictions_area.tolist()
        latencys = predictions_latency.tolist()
        acc = cal_acc(areas,latencys,samples_y[test_index])
        models_acc[name] = acc
        predict_results[name]["area_predictions"]=areas
        predict_results[name]["latency_predictions"] = latencys
        predict_results[name]["acc"] = acc
    return models_acc,predict_results