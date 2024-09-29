import numpy as np
def encode(x):
    original_feature = extract(x)
    x_encode,discretized_feature= convert(x,original_feature)
    return x_encode,discretized_feature,original_feature

def extract(x):
    directive_choices = []
    for i in range(x.shape[1]):
        unique_values = np.unique(x[:,i]).tolist()
        directive_choices.append(unique_values)
    return directive_choices
def convert(X,featureSets):
    discretized_feature = discretize_dataset(featureSets)
    d = np.empty((X.shape[0],X.shape[1]))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            index = featureSets[j].index(X[i][j])
            try:
                d[i,j] = discretized_feature[j][index]
            except Exception as e:
                print()
    X = d
    discretized_feature = np.array(discretized_feature,dtype=object)
    return X,discretized_feature
def discretize_dataset( lattice_descriptor):
    discretized_feature = []
    for feature in lattice_descriptor:
        # tmp = []
        # for x in frange(0, 1, 1. / (len(feature) - 1)):
        #     tmp.append(x)
        tmp = np.linspace(0,1,len(feature))
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