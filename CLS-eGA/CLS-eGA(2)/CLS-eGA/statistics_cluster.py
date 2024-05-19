import numpy as np #导入矩阵处理库
import scipy.io as scio
from statistics_utils import *
# data=scio.loadmat('D:/paper code/ClusterBasedDSE/Copy_of_ClusterBasedDSE_MatlabCode/explorations_results/contrastive/timed_add_bias_to_activations_backprop_backprop_expN50_prob_iS15_cF15.mat')
# python_y=np.array(data['matlab_y']) #将matlab数据赋值给python变量
benchNames = [
    "bulk",
    "md_kernel_knn_md",
    "viterbi_viterbi_viterbi",
    "ncubed",
    "bbgemm_blocked_gemm",
    "fft_strided_fft",
    "twiddles8_transpose_fft",
    "ms_mergesort_merge_sort",
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
    'ellpack_ellpack_spmv',
    'merge_merge_sort',
    'stencil3d_stencil3d_stencil',
    'get_oracle_activations1_backprop_backprop',
    'get_oracle_activations2_backprop_backprop',
    'matrix_vector_product_with_bias_input_layer',
    'matrix_vector_product_with_bias_output_layer',
    'matrix_vector_product_with_bias_second_layer',
    'backprop_backprop_backprop',
    'add_bias_to_activations_backprop_backprop',
    'soft_max_backprop_backprop',
    'take_difference_backprop_backprop',
    'update_weights_backprop_backprop',
    'get_delta_matrix_weights1',
    'get_delta_matrix_weights2',
    'get_delta_matrix_weights3',
    'stencil_stencil2d_stencil'

]
design_space_size = [
    2352,
    1600,
    1152,
    2744,
    1600,
    1600,
    64,
    1024,
    2400,
    4704,
    484,
    1280,
    1600,
    704,
    1792,
    500,
    50,
    625,
    20,
    18,
    216,
    1944,
    1600,
    4096,
    1536,
    2401,
    1372,
    1372,
    392,
    686,
    2048,
    1372,
    64,
    512,
    1024,
    21952,
    31213,
    21952,
    336



]
types = ["contrastive","Origin"]
import pandas as pd
import pathlib
df = pd.DataFrame(columns=["benchmark","type","mean_adrs","mean_time","start_synth","mean_synth","evolution_synth"])
to_df = pd.DataFrame(columns=["benchmark",
                  "CL-Cluster",
                  "Cluster",
                  "difference",
                  "CL-Cluster",
                  "Cluster",
                    "start",
                  "CL-Cluster",
                  "Cluster",
                  "Time Saved",
                  "Cl-Cluster Time",
                  "Cluster time",
                  "Time Saved",
                  "CL-Cluster Paretos",
                  "Cluster Paretos"
                  ])
def get_file_paths(benchName,n_runs,type):
    file = 'D:/paper code/ClusterBasedDSE/Copy_of_ClusterBasedDSE_MatlabCode/explorations_results/' + type + '/timed_' + benchName + '_expN'+str(n_runs)+'_prob_iS5_cF5.mat'
    return file

from statistics_utils import calculate_cluster_mean_time
for i,benchName in enumerate(benchNames):
    total_df = {"benchmark": benchName,
                  "contrastive_mean_adrs": None,
                  "Origin_mean_adrs": None,
                  "ADRS Difference":None,
                  "contrastive_mean_time": None,
                  "Origin_mean_time": None,
                  "start": None,
                  "contrastive_mean_synth": None,
                  "Origin_mean_synth": None,
                  "save synthesis time": None,
                  "contrastive_total_time": None,
                  "Origin_total_time": None,
                  "save total time": None,
                  "contrastive_paretos":None,
                  "origin_paretos":None
                  }

    save_synthesis_time = None
    contrastive_total_time = None
    Origin_total_time = None
    save_total_time = None
    start = design_space_size[i]*0.05
    n_runs = 50
    file = get_file_paths(benchName,n_runs,"contrastive")
    if not pathlib.Path(file).exists():
        n_runs = 20
        file = get_file_paths(benchName, n_runs, "contrastive")
        if not pathlib.Path(file).exists():
            n_runs = 10
            file = get_file_paths(benchName, n_runs, "contrastive")
            if not pathlib.Path(file).exists():
                continue
    contrastive_synthesis_time = 0
    origin_synthesis_time = 0
    contrastive_paretos = 0
    origin_paretos = 0
    for type in types:
        file = get_file_paths(benchName,n_runs,type)
        data = scio.loadmat(file)
        n = data["nOfExplorations"][0][0]
        start_synth = round(design_space_size[i] * data["percentage"][0][0] / 100)
        synEvolution = data["synthEvolution"].tolist()[0]
        mean_synth = sum([synEvolution[i][0][0] for i in range(len(synEvolution))])/n
        adrses = data["finalADRS"].tolist()
        print(benchName)
        mean_adrs = sum([adrs[0] for adrs in adrses])/n
        total_df[type+"_mean_adrs"] = mean_adrs
        time = data["timeEvolutions"].tolist()[0]
        mean_time = sum([time[i][0][0] for i in range(len(time))]) / n
        evolution_synth = mean_synth - start_synth
        df.loc[len(df.index)] = [benchName,type,mean_adrs,mean_time,start_synth,mean_synth,evolution_synth]
        total_df[type + "_mean_adrs"] = mean_adrs
        total_df[type + "_mean_time"] = mean_time
        total_df[type + "_mean_synth"] = mean_synth-start
        if type == "contrastive":
            contrastive_synthesis_time = calculate_cluster_mean_time(data, benchName, n)
            contrastive_paretos = compare_cluster_pareto_points(data, benchName, n)

        else:
            origin_synthesis_time = calculate_cluster_mean_time(data, benchName, n)
            origin_paretos = compare_cluster_pareto_points(data,benchName,n)
    total_df["save synthesis time"] = origin_synthesis_time - contrastive_synthesis_time
    total_df["contrastive_total_time"] = total_df["contrastive_mean_time"] + contrastive_synthesis_time
    total_df["start"] = round(design_space_size[i]*0.05)
    total_df["Origin_total_time"] = total_df["Origin_mean_time"] + origin_synthesis_time
    total_df["save total time"] = total_df["Origin_total_time"] - total_df["contrastive_total_time"]
    total_df["ADRS Difference"] = total_df["contrastive_mean_adrs"] - total_df["Origin_mean_adrs"]
    total_df["contrastive_paretos"] = contrastive_paretos
    total_df["origin_paretos"] = origin_paretos
    to_df.loc[len(to_df.index)] = total_df.values()
df.to_csv("statistics/cluster.csv",index=False)
to_df.to_csv("statistics/total_cluster.csv",index=False)
