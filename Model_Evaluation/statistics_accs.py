import copy
import os

import numpy as np
import pandas as pd
all_benchmarks = [
    "ellpack_ellpack_spmv",
    "bulk",
    "md_kernel_knn_md",
    "viterbi_viterbi_viterbi",
    "ncubed",
    "bbgemm_blocked_gemm",
    "fft_strided_fft",
    "twiddles8_transpose_fft",
    "ms_mergesort_merge_sort",
    "merge_merge_sort",
    'stencil_stencil2d_stencil',
    'stencil3d_stencil3d_stencil',
    "update_radix_sort",
    "hist_radix_sort",
    "init_radix_sort",
    "sum_scan_radix_sort",
    "last_step_scan_radix_sort",
    "local_scan_radix_sort",
    "ss_sort_radix_sort",
    'aes_addRoundKey_aes_aes',
    'aes_subBytes_aes_aes',
    'aes_addRoundKey_cpy_aes_aes',
    'aes_shiftRows_aes_aes',
    'aes_mixColumns_aes_aes',
    'aes_expandEncKey_aes_aes',
    'aes256_encrypt_ecb_aes_aes',
    'get_oracle_activations1_backprop_backprop',
    "get_oracle_activations2_backprop_backprop",
    'matrix_vector_product_with_bias_input_layer',
    'matrix_vector_product_with_bias_second_layer',
    'matrix_vector_product_with_bias_output_layer',
    'backprop_backprop_backprop',
    'add_bias_to_activations_backprop_backprop',
    'soft_max_backprop_backprop',
    'take_difference_backprop_backprop',
    'update_weights_backprop_backprop',
    'get_delta_matrix_weights1',
    'get_delta_matrix_weights2',
    'get_delta_matrix_weights3',
]
design_space_size = {
'ellpack_ellpack_spmv': 1600,
 'bulk': 2352,
 'md_kernel_knn_md': 1600,
 'viterbi_viterbi_viterbi': 1152,
 'ncubed': 2744,
 'bbgemm_blocked_gemm': 1600,
 'fft_strided_fft': 1600,
 'twiddles8_transpose_fft': 64,
 'ms_mergesort_merge_sort': 1024,
 'merge_merge_sort': 4096,
 'stencil_stencil2d_stencil': 336,
 'stencil3d_stencil3d_stencil': 1536,
 'update_radix_sort': 2400,
 'hist_radix_sort': 4704,
 'init_radix_sort': 484,
 'sum_scan_radix_sort': 1280,
 'last_step_scan_radix_sort': 1600,
 'local_scan_radix_sort': 704,
 'ss_sort_radix_sort': 1792,
 'aes_addRoundKey_aes_aes': 500,
 'aes_subBytes_aes_aes': 50,
 'aes_addRoundKey_cpy_aes_aes': 625,
 'aes_shiftRows_aes_aes': 20,
 'aes_mixColumns_aes_aes': 18,
 'aes_expandEncKey_aes_aes': 216,
 'aes256_encrypt_ecb_aes_aes': 1944,
 'get_delta_matrix_weights1': 21952,
 'get_delta_matrix_weights2': 31213,
 'get_delta_matrix_weights3': 21952,
 'get_oracle_activations1_backprop_backprop': 2401,
 'get_oracle_activations2_backprop_backprop': 1372,
 'matrix_vector_product_with_bias_input_layer': 1372,
 'matrix_vector_product_with_bias_second_layer': 686,
 'matrix_vector_product_with_bias_output_layer': 392,
 'backprop_backprop_backprop': 2048,
 'add_bias_to_activations_backprop_backprop': 1372,
 'soft_max_backprop_backprop': 64,
 'take_difference_backprop_backprop': 512,
 'update_weights_backprop_backprop': 1024
}

selected_contrastive_models = [
    "RandomForestClassifier",
    "XGBClassifier",
    "ExtraTreesClassifier",
    "BaggingClassifier",
    "GradientBoostingClassifier",
    "AdaBoostClassifier",
    "HistGradientBoostingClassifier",
    "LGBMClassifier",
    "DecisionTreeClassifier",
    "ExtraTreeClassifier",
    "DummyClassifier",
    # "GaussianProcessClassifier",
    "LogisticRegression",
    "LogisticRegressionCV",
    "KNeighborsClassifier",
     "BernoulliNB",
     "GaussianNB",
    "MLPClassifier"
]
regression_models = [
 'ExtraTreesRegressor',
 # 'NuSVR',
 'HuberRegressor',
 'BaggingRegressor',
 'RandomForestRegressor',
 'GaussianProcessRegressor',
 'Lars',
 'LinearRegression',
 'TransformedTargetRegressor',
 'RidgeCV',
 'Ridge',
 'KNeighborsRegressor',
 'BayesianRidge',
 'LarsCV',
 'LassoLarsCV',
 'LassoCV',
 'GammaRegressor',
 'ElasticNetCV',
 'SGDRegressor',
 'LinearSVR',
 # 'OrthogonalMatchingPursuitCV',
 'GradientBoostingRegressor',
 'LassoLarsIC',
 'RANSACRegressor',
 'XGBRegressor',
 'DecisionTreeRegressor',
 'TweedieRegressor',
 'QuantileRegressor',
 'OrthogonalMatchingPursuit',
 'AdaBoostRegressor',
 'PoissonRegressor',
 'LassoLars',
 'Lasso',
 'HistGradientBoostingRegressor',
 'ElasticNet',
 'DummyRegressor',
 'LGBMRegressor',
 'ExtraTreeRegressor',
 'KernelRidge',
 'PassiveAggressiveRegressor',
 'MLPRegressor',
 # 'SVR'
]
print("Total Benchmarks：%d" % len(all_benchmarks))
ignore =  [
    "get_delta_matrix_weights1"

]
all_model_symbols = dict()
all_models = copy.deepcopy(selected_contrastive_models)
all_models.extend(regression_models)
index = 1
for classifier in selected_contrastive_models:
    all_model_symbols[classifier] = "c"+str(index)
    index +=1
for reg in regression_models:
    all_model_symbols[reg] = "r" + str(index)
    index += 1
models_acc = dict.fromkeys(all_models, 0)

path = "./results/"
files = os.listdir(path)
benchmarks_acc = dict()
test_benchmarks = []
print("\033[0;32;40mTEST BENCHMARKS：\033[0m")
for f in files:
    if f[-8:] != "_acc.npy":
        continue
    benchmark_name = f[:-8]
    acc =  np.load(path+f,allow_pickle=True).item()
    benchmarks_acc[benchmark_name] = acc[benchmark_name]
    test_benchmarks.append(benchmark_name)
    print(benchmark_name,design_space_size[benchmark_name])
print("\033[0;32;40mTotal：%d\033[0m" % len(test_benchmarks))
un_test_benchmarks = list(set(all_benchmarks) - set(test_benchmarks))
print("\033[0;31;40mUN TEST BENCHMARKS\033[0m")
for b in un_test_benchmarks:
    print(b,design_space_size[b])
print("\033[0;31;40mTotal：%d\033[0m" % len(un_test_benchmarks))
sorts = []
total = 0
model_all_accs = {}
benchmarks = []
for benchmark, con_reg_models in benchmarks_acc.items():

    benchmarks.append(benchmark)
    contrastive_models = con_reg_models["contrastive"]
    regression_models = con_reg_models["regression"]
    total += 1
    benchmark_sort = copy.deepcopy(models_acc)
    for con_model_name, acc in contrastive_models.items():
        if con_model_name not in model_all_accs:
            model_all_accs[con_model_name] = []
        benchmark_sort[con_model_name] = acc
        models_acc[con_model_name]+=(acc)
        model_all_accs[con_model_name].append(acc)
    for reg_model_name, acc in regression_models.items():
        if reg_model_name not in model_all_accs:
            model_all_accs[reg_model_name] = []
        benchmark_sort[reg_model_name] = acc
        models_acc[reg_model_name]+=(acc)
        model_all_accs[reg_model_name].append(acc)
    sorts.append(sorted(benchmark_sort.items(), key=lambda x: x[1], reverse=True))
model_stds = dict()
for key,value in model_all_accs.items():
    model_stds[key] = np.std(value,ddof=1)
for model_name,total_acc in models_acc.items():
    models_acc[model_name]/=total
d_order=sorted(models_acc.items(),key=lambda x:x[1],reverse=True)
models_acc_sort = dict()
for d in d_order:
    models_acc_sort[d[0]] = d[1]
rank = 1
for m in d_order:
    if m[0] in selected_contrastive_models:
        print("\033[0;32;40m%d %s：%f\033[0m" % (rank,m[0],m[1]))
    else:
        print("\033[0;31;40m%d %s：%f\033[0m" % (rank,m[0],m[1]))
    rank+=1
top = 3
top_classifier = []
top_regressor = []
least_classifier = []
least_regressor = []
print("For each benchmark, top=%d models with the highest accuracy are selected" % top)
for index,model_ranks in enumerate(sorts):
    print("\033[0;33;40m %s \033[0m" % (files[index][:-8]))
    rank =1
    flagc,flagr = 1,1
    for i in range(top):
        name = model_ranks[i][0]
        if rank == 1 and name in selected_contrastive_models:
            top_classifier.append(files[index][:-8])
        if rank == 1 and name in regression_models:
            top_regressor.append(files[index][:-8])
        if flagc and name in selected_contrastive_models:
            least_classifier.append(files[index][:-8])
            flagc = 0
        if flagr and name in regression_models:
            flagr = 0
            least_regressor.append(files[index][:-8])

        if name in selected_contrastive_models:
            print("\033[0;32;40m%d %s: %f\033[0m" % (rank,name,model_ranks[i][1]))
        else:
            print("\033[0;31;40m%d %s: %f\033[0m" % (rank,name, model_ranks[i][1]))
        rank+=1
    print()

print("\033[0;33;40m Total: %d \033[0m" % (len(sorts)))
print("\033[0;32;40m Highest accuracy model is classifier \033[0m")
for m in top_classifier:
    print(m)
print("\033[0;32;40m total：%d\033[0m"%len(top_classifier))
print("\033[0;31;40m Highest accuracy model is regressor \033[0m")
for m in top_regressor:
    print(m)
print("\033[0;31;40m total：%d\033[0m"%len(top_regressor))

print("\033[0;32;40m At least one classifier ranked in the top=%d\033[0m"%top)
for m in least_classifier:
    print(m)
print("\033[0;32;40m total：%d\033[0m"%len(least_classifier))
print("\033[0;31;40m At least one regressor ranked in the top=%d\033[0m"%top)
for m in least_regressor:
    print(m)
print("\033[0;31;40m total：%d\033[0m"%len(least_regressor))
models_acc_sort_copy = pd.DataFrame(columns =["Model","Accuracy","Type"])
print("\033[0;32;40m model symbols \033[0m")
for key,value in models_acc_sort.items():
    symbol = all_model_symbols[key]
    t = "classifier" if symbol.startswith("c") else "regressor"
    models_acc_sort_copy.loc[len(models_acc_sort_copy.index)] = [key+": "+symbol,value,t]
    print(symbol,key,value)
models_acc_sort_copy.to_csv("./statictis/models_sort_accuracy.csv",index=False)


benchmarks = {
    "spmv ellpack":["ellpack_ellpack_spmv"],
    "bfs bulk":["bulk"],
    "md knn":["md_kernel_knn_md"],
    "viterbi":["viterbi_viterbi_viterbi"],
    "gemm":["ncubed",
    "bbgemm_blocked_gemm"],
    "fft":["fft_strided_fft",
        "twiddles8_transpose_fft"],
    "sort merge":[
        "ms_mergesort_merge_sort",
        "merge_merge_sort"],
    "stencil":[
        'stencil_stencil2d_stencil',
        'stencil3d_stencil3d_stencil',
        ],
    "radix sort":["update_radix_sort",
        "hist_radix_sort",
        "init_radix_sort",
        "sum_scan_radix_sort",
        "last_step_scan_radix_sort",
        "local_scan_radix_sort",
        "ss_sort_radix_sort",
                    ],
    "aes":['aes_addRoundKey_aes_aes',
            'aes_subBytes_aes_aes',
            'aes_addRoundKey_cpy_aes_aes',
            'aes_shiftRows_aes_aes',
            'aes_mixColumns_aes_aes',
            'aes_expandEncKey_aes_aes',
            'aes256_encrypt_ecb_aes_aes',
           ],
    "backprop":[
        'get_delta_matrix_weights1',
        'get_delta_matrix_weights2',
        'get_delta_matrix_weights3',
        'get_oracle_activations1_backprop_backprop',
        "get_oracle_activations2_backprop_backprop",
        'matrix_vector_product_with_bias_input_layer',
        'matrix_vector_product_with_bias_second_layer',
        'matrix_vector_product_with_bias_output_layer',
        'backprop_backprop_backprop',
        'add_bias_to_activations_backprop_backprop',
        'soft_max_backprop_backprop',
        'take_difference_backprop_backprop',
        'update_weights_backprop_backprop',
    ]
}

sorts = []
for key in benchmarks.keys():
    sum_benchmark = dict.fromkeys(models_acc.keys(),0)
    total = 0
    for sub in benchmarks[key]:
        if sub in benchmarks_acc:
            total +=1
    for sub in benchmarks[key]:
        if sub not in benchmarks_acc:
            continue
        classifier_accs = benchmarks_acc[sub]["contrastive"]
        reg_accs = benchmarks_acc[sub]["regression"]
        for name,acc in classifier_accs.items():
            sum_benchmark[name]+=acc/total
        for name,acc in reg_accs.items():
            sum_benchmark[name]+=acc/total
    sorts.append(sorted(sum_benchmark.items(), key=lambda x: x[1], reverse=True))
df3 = pd.DataFrame(columns=["classifier","benvhmark","acc","type"])
benchmark_names = list(benchmarks.keys())
top = 3
for index,benchmark in enumerate(benchmark_names):
    r = sorts[index][0:top]
    ranks = sorted(r,key=lambda x:x[1],reverse=True)
    column = []
    for item in ranks:
        if all_model_symbols[item[0]].startswith("c"):
            df3.loc[len(df3.index)] = [all_model_symbols[item[0]],benchmark,item[1],"classifier"]
        else:
            df3.loc[len(df3.index)] = [all_model_symbols[item[0]], benchmark, item[1], "regressor"]
df3.to_csv("./statictis/benchmark_sort.csv",index=False)
print()

model_id = pd.DataFrame(columns=["ID","Model","ID","Model"])
temp = []
count = 0
for key,value in all_model_symbols.items():
    temp.append(value)
    temp.append(key)
    count += 1
    if count == 2:
        model_id.loc[len(model_id.index)] = temp
        count = 0
        temp = []
model_id.to_csv("./statictis/model_symbols.csv",index=False)











