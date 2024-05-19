import numpy as np #导入矩阵处理库
import scipy.io as scio
data=scio.loadmat('D:/paper code/ClusterBasedDSE/Copy_of_ClusterBasedDSE_MatlabCode/explorations_results/contrastive/timed_add_bias_to_activations_backprop_backprop_expN50_prob_iS15_cF15.mat')
# python_y=np.array(data['matlab_y']) #将matlab数据赋值给python变量
benchNames = [
# 'add_bias_to_activations_backprop_backprop',
# 'aes_addRoundKey_aes_aes',
# 'aes_addRoundKey_cpy_aes_aes',
# 'aes_expandEncKey_aes_aes',
# 'aes_table',
# 'aes256_encrypt_ecb_aes_aes',
# 'backprop_backprop_backprop',
# 'ellpack_ellpack_spmv',
# 'get_oracle_activations2_backprop_backprop',
# 'matrix_vector_product_with_bias_input_layer',
# 'matrix_vector_product_with_bias_output_layer',
# 'matrix_vector_product_with_bias_second_layer',
#     'merge_merge_sort',
#     'take_difference_backprop_backprop',
#     'update_weights_backprop_backprop',

# "bulk",
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
    # 'aes_subBytes_aes_aes',
'aes_addRoundKey_cpy_aes_aes',
# 'aes_shiftRows_aes_aes',
#    'aes_mixColumns_aes_aes',
    'aes_expandEncKey_aes_aes',
    'aes256_encrypt_ecb_aes_aes'

]
design_space_size = [
    1372,
    500,
    625,
    216,
    3072,
    1944,
    2048,
    1600,
    1372,
    1372,
    392,
    686,
    4096,
    512,
    1024,
    1600,
1152,
    2744,
    1600,
1600,
    64,
    1024,
2400,
    1600,
704,
1792,
2352



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
                  "Time Saved",])
mean_synthesize_time = {
'add_bias_to_activations_backprop_backprop':25.7,
'aes_addRoundKey_aes_aes':31.1,
'aes_addRoundKey_cpy_aes_aes':32.2,
'aes_expandEncKey_aes_aes':33.6,
'aes_notable':35.4,
'aes_table':33.9,
'aes256_encrypt_ecb_aes_aes':103.7,
'backprop_backprop_backprop':53,
'ellpack_ellpack_spmv':85.9,
'get_oracle_activations2_backprop_backprop':38.6,
'matrix_vector_product_with_bias_input_layer':62.5,
'matrix_vector_product_with_bias_output_layer':22.5,
'matrix_vector_product_with_bias_second_layer':33.4,
'merge_merge_sort':99.4,
'take_difference_backprop_backprop':18.1,
'update_weights_backprop_backprop':46.3,
                       "md_kernel_knn_md":46.3,
                       "viterbi_viterbi_viterbi":46.3,
                       "ncubed":46.3,
                       "bbgemm_blocked_gemm":46.3,
                       "fft_strided_fft":46.3,
                       "twiddles8_transpose_fft":46.3,
                       "ms_mergesort_merge_sort":46.3,
                       "update_radix_sort":46.3,
                       "hist_radix_sort":46.3,
                       "init_radix_sort":46.3,
                       "sum_scan_radix_sort":46.3,
                       "last_step_scan_radix_sort":46.3,
                       "local_scan_radix_sort":46.3,
                       "ss_sort_radix_sort":46.3,
}

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
                  }
    n = data["nOfExplorations"][0][0]
    start_synth = round(design_space_size[i] * data["percentage"][0][0] / 100)

    save_synthesis_time = None
    contrastive_total_time = None
    Origin_total_time = None
    save_total_time = None
    start = design_space_size[i]*0.15
    for type in types:
        file = 'D:/paper code/ClusterBasedDSE/Copy_of_ClusterBasedDSE_MatlabCode/explorations_results/'+type+'/timed_'+benchName+'_expN50_prob_iS15_cF15.mat'
        if not pathlib.Path(file).exists():
            break

        data = scio.loadmat(
            'D:/paper code/ClusterBasedDSE/Copy_of_ClusterBasedDSE_MatlabCode/explorations_results/'+type+'/timed_'+benchName+'_expN50_prob_iS15_cF15.mat')
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
    synthesis_time = mean_synthesize_time[benchName]
    total_df["save synthesis time"] = synthesis_time * (total_df["Origin_mean_synth"] - total_df["contrastive_mean_synth"])
    total_df["contrastive_total_time"] = total_df["contrastive_mean_time"] + synthesis_time * total_df[
        "contrastive_mean_synth"]
    total_df["start"] = round(design_space_size[i]*0.15)
    total_df["Origin_total_time"] = total_df["Origin_mean_time"] + synthesis_time * total_df["Origin_mean_synth"]
    total_df["save total time"] = total_df["Origin_total_time"] - total_df["contrastive_total_time"]
    total_df["ADRS Difference"] = total_df["contrastive_mean_adrs"] - total_df["Origin_mean_adrs"]
    to_df.loc[len(to_df.index)] = total_df.values()
df.to_csv("statistics/cluster.csv",index=False)
# to_df['Col_sum'] = to_df.apply(lambda x: x.sum(), axis=1)
to_df.to_csv("statistics/total_cluster.csv",index=False)
