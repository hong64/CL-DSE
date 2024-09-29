from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from datasets import get_dse_description
from sklearn.model_selection import KFold
from sklearn.ensemble import *
from xgboost import XGBClassifier, XGBRegressor
from models_classification import *
from models_regresssion import Regression_Models
benchmarks = [
    # "ellpack_ellpack_spmv",
    # "bulk",
    # "md_kernel_knn_md",
    # "viterbi_viterbi_viterbi",
    # "ncubed",
    # "bbgemm_blocked_gemm",
    # "fft_strided_fft",
    # # "twiddles8_transpose_fft",
    # "ms_mergesort_merge_sort",
    # "merge_merge_sort",
    'stencil_stencil2d_stencil',
    # 'stencil3d_stencil3d_stencil',
    # "update_radix_sort",
    # "hist_radix_sort",
    # "init_radix_sort",
    # "sum_scan_radix_sort",
    # "last_step_scan_radix_sort",
    # "local_scan_radix_sort",
    # "ss_sort_radix_sort",
    # 'aes_addRoundKey_aes_aes',
    # 'aes_subBytes_aes_aes',
    # 'aes_addRoundKey_cpy_aes_aes',
    # 'aes_shiftRows_aes_aes',
    # 'aes_mixColumns_aes_aes',
    # 'aes_expandEncKey_aes_aes',
    # 'aes256_encrypt_ecb_aes_aes',
    # 'get_oracle_activations1_backprop_backprop',
    # "get_oracle_activations2_backprop_backprop",
    # 'matrix_vector_product_with_bias_input_layer',
    # 'matrix_vector_product_with_bias_second_layer',
    # 'matrix_vector_product_with_bias_output_layer',
    # 'backprop_backprop_backprop',
    # 'add_bias_to_activations_backprop_backprop',
    # 'soft_max_backprop_backprop',
    # 'take_difference_backprop_backprop',
    # 'update_weights_backprop_backprop',
    # 'get_delta_matrix_weights1',
    # 'get_delta_matrix_weights2',
    # 'get_delta_matrix_weights3',
]



class Sample():
    def __init__(self,samples_x,samples_y, train_index, test_index, d):
        self.x_train = samples_x[train_index]
        self.y_train = samples_y[train_index]
        self.x_test = samples_x[test_index]
        self.y_test = samples_y[test_index]
        self.train_index = train_index
        self.test_index = test_index
        self.d = d
def _average(target_dict,acc_dict,N):
    for name, acc in acc_dict.items():
        target_dict[name] += (acc / N)
    return target_dict

from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import *
from sklearn.dummy import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import *
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
selected_contrastive_models = {
    "RandomForestClassifier":RandomForestClassifier(),
    "XGBClassifier":XGBClassifier(),
    "ExtraTreesClassifier":ExtraTreesClassifier(),
    "BaggingClassifier":BaggingClassifier(),
    "GradientBoostingClassifier":GradientBoostingClassifier(),
    "AdaBoostClassifier":AdaBoostClassifier(),
    "HistGradientBoostingClassifier":HistGradientBoostingClassifier(),
    "LGBMClassifier":LGBMClassifier(),
    "DecisionTreeClassifier":DecisionTreeClassifier(),
    "ExtraTreeClassifier":ExtraTreeClassifier(),
    "DummyClassifier":DummyClassifier(),
    "LogisticRegression":LogisticRegression(),
    "LogisticRegressionCV":LogisticRegressionCV(),
    "KNeighborsClassifier":KNeighborsClassifier(),
     "BernoulliNB":BernoulliNB(),
     "GaussianNB":GaussianNB(),
    "MLPClassifier":MLPClassifier()

}
from sklearn.ensemble import *
from sklearn.tree import *
regression_models = {
'ExtraTreesRegressor':ExtraTreesRegressor(),
 'HuberRegressor':HuberRegressor(),
 'BaggingRegressor':BaggingRegressor(),
 'RandomForestRegressor':RandomForestRegressor(),
 'GaussianProcessRegressor':GaussianProcessRegressor(),
 'Lars':Lars(),
 'LinearRegression':LinearRegression(),
 'TransformedTargetRegressor':TransformedTargetRegressor(),
 'RidgeCV':TransformedTargetRegressor(),
 'Ridge':Ridge(),
 'KNeighborsRegressor':KNeighborsRegressor(),
 'BayesianRidge':BayesianRidge(),
 'LarsCV':LarsCV(),
 'LassoLarsCV':LassoLarsCV(),
 'LassoCV':LassoCV(),
 'GammaRegressor':GammaRegressor(),
 'ElasticNetCV':ElasticNetCV(),
 'SGDRegressor':SGDRegressor(),
 'LinearSVR':LinearSVR(),
 'GradientBoostingRegressor':GradientBoostingRegressor(),
 'LassoLarsIC':LassoLarsIC(),
 'RANSACRegressor':RANSACRegressor(),
 'XGBRegressor':XGBRegressor(),
 'DecisionTreeRegressor':DecisionTreeRegressor(),
 'TweedieRegressor':TweedieRegressor(),
 'QuantileRegressor':QuantileRegressor(),
 'OrthogonalMatchingPursuit':OrthogonalMatchingPursuit(),
 'AdaBoostRegressor':AdaBoostRegressor(),
 'PoissonRegressor':PoissonRegressor(),
 'LassoLars':LassoLars(),
 'Lasso':Lasso(),
 'HistGradientBoostingRegressor':HistGradientBoostingRegressor(),
 'ElasticNet':ElasticNet(),
 'DummyRegressor':DummyRegressor(),
 'LGBMRegressor':LGBMRegressor(),
 'ExtraTreeRegressor':ExtraTreeRegressor(),
 'KernelRidge':KernelRidge(),
 'PassiveAggressiveRegressor':PassiveAggressiveRegressor(),
 'MLPRegressor':MLPRegressor(),
}

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


selected_regression_models = dict.fromkeys(regression_models.keys(),None)
benchmark_dict = dict()
benchmark_samples = dict()
error_benchmarks = dict()
for benchmark in benchmarks:
    benchmark_dict[benchmark] = copy.deepcopy({"contrastive":None,"regression":None})
for i,benchmark in enumerate(benchmarks):
    print("-----------------------------------%s-------------------------------------------------"%benchmark)
    sample_size = round(design_space_size[benchmark]*0.05)
    x, y, status, discretized_feature, original_feature = get_dse_description(benchmark)
    K_Fold,iterations = 5,20
    contrastive_iter_acc = dict.fromkeys(selected_contrastive_models.keys(), 0)
    regression_iter_acc = dict.fromkeys(selected_regression_models.keys(), 0)
    persist = {iteration:{} for iteration in range(iterations)}
    j = 0
    # for j in range(iterations):
    while j < iterations:
        sample_ids = np.random.choice(x.shape[0],sample_size,replace=False).tolist()
        x_sample,y_sample = x[sample_ids,:],y[sample_ids]
        success_ids = np.where((y_sample != [10000000, 10000000]).all(1))[0].tolist()
        samples_x = x_sample[success_ids, :]
        samples_y = y_sample[success_ids, :]
        kf = KFold(n_splits=K_Fold, shuffle=False)  # 初始化KFold
        cla_k_acc = dict.fromkeys(selected_contrastive_models.keys(), 0)
        reg_k_acc = dict.fromkeys(selected_regression_models.keys(), 0)
        index = 1
        persist[j]["x"] = samples_x
        persist[j]["y"] = samples_y
        try:
            for train_index, test_index in kf.split(samples_x):
                samples = Sample(samples_x,samples_y, train_index, test_index, x.shape[1])
                reg_model_accs,reg_model_predict = Regression_Models(selected_regression_models, samples_x, samples_y, train_index, test_index,regression_models)
                cla_model_accs,con_model_predicts = Contrastive_Models(samples, selected_contrastive_models)
                cla_k_acc = _average(cla_k_acc, cla_model_accs, K_Fold)
                reg_k_acc = _average(reg_k_acc, reg_model_accs, K_Fold)
                persist[j][index] = {"train_index":train_index, "test_index":test_index, "contrastive":con_model_predicts, "regression":reg_model_predict}
                index+=1
        except Exception as e:
            continue
        j+=1
        contrastive_iter_acc = _average(contrastive_iter_acc, cla_k_acc, iterations)
        regression_iter_acc = _average(regression_iter_acc, reg_k_acc, iterations)
    benchmark_samples[benchmark] = persist
    benchmark_dict[benchmark]["contrastive"] = contrastive_iter_acc
    benchmark_dict[benchmark]["regression"] = regression_iter_acc
    np.save("./results/"+benchmark, benchmark_samples)
    np.save('./results/'+benchmark+'_acc', benchmark_dict)
np.save('model_selection',benchmark_dict)
np.save("model_records",benchmark_samples)
np.save("error_benchmarks",error_benchmarks)




