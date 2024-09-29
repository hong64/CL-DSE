import copy

from ReadDB import  readDB
import numpy as np
import pandas as pd

benchmarks = [
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
# 'merge_merge_sort',
# 'take_difference_backprop_backprop',
# 'update_weights_backprop_backprop',
# "bulk"
# "viterbi_viterbi_viterbi"
# 'aes_notable',
# "bulk",
# "md_kernel_knn_md",
# "viterbi_viterbi_viterbi",
# "ncubed",
# "bbgemm_blocked_gemm",
# "fft_strided_fft",
# "twiddles8_transpose_fft",
# "ms_mergesort_merge_sort",
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
# 'aes256_encrypt_ecb_aes_aes'
]

mean_synthesize_time = {
# 'add_bias_to_activations_backprop_backprop':25.7,
# 'aes_addRoundKey_aes_aes':31.1,
# 'aes_addRoundKey_cpy_aes_aes':32.2,
# 'aes_expandEncKey_aes_aes':33.6,
# 'aes_notable':35.4,
# 'aes_table':33.9,
# 'aes256_encrypt_ecb_aes_aes':103.7,
# 'backprop_backprop_backprop':53,
# 'ellpack_ellpack_spmv':85.9,
# 'get_oracle_activations2_backprop_backprop':38.6,
# 'matrix_vector_product_with_bias_input_layer':62.5,
# 'matrix_vector_product_with_bias_output_layer':22.5,
# 'matrix_vector_product_with_bias_second_layer':33.4,
'merge_merge_sort':99.4,
# 'take_difference_backprop_backprop':18.1,
# 'update_weights_backprop_backprop':46.3,
                           "bulk":0,
                       "md_kernel_knn_md":0,
                       "viterbi_viterbi_viterbi":0,
                       "ncubed":0,
                       "bbgemm_blocked_gemm":0,
                       "fft_strided_fft":0,
                       "twiddles8_transpose_fft":0,
                       "ms_mergesort_merge_sort":0,
                       "update_radix_sort":0,
                       "hist_radix_sort":0,
                       "init_radix_sort":0,
                       "sum_scan_radix_sort":0,
                       "last_step_scan_radix_sort":0,
                       "local_scan_radix_sort":0,
                       "ss_sort_radix_sort":0,
                       'aes_addRoundKey_aes_aes':0,
                       'aes_subBytes_aes_aes':0,
                       'aes_addRoundKey_cpy_aes_aes':0,
                       'aes_shiftRows_aes_aes':0,
                       'aes_mixColumns_aes_aes':0,
                       'aes_expandEncKey_aes_aes':0,
                       'aes256_encrypt_ecb_aes_aes':0
}
import pathlib
ini_synth = 20
items = ["final_adrs","max_n_of_synth","timeEvolutions"]
csv_items = ["mean_adrs","mean_synth","mean_time"]
df = pd.DataFrame(columns=["benchmark","type","mean_adrs","mean_time","start_synth","mean_synth","evolution_synth"])
total_df = pd.DataFrame(columns=["benchmark",
                  "contrastive_mean_adrs",
                  "Origin_mean_adrs",
                  "ADRS Difference",
                  "contrastive_mean_time",
                  "Origin_mean_time",
                  "start",
                  "contrastive_mean_synth",
                  "Origin_mean_synth",
                  "save synthesis time",
                  "contrastive_total_time",
                  "Origin_total_time",
                  "save total time",])
for benchmark in benchmarks:
    Total_dict = {"benchmark":benchmark,
                  "contrastive_mean_adrs":None,
                  "Origin_mean_adrs":None,
                  "ADRS Difference":None,
                  "contrastive_mean_time":None,
                  "Origin_mean_time":None,
                  "start":None,
                  "contrastive_mean_synth":None,
                  "Origin_mean_synth":None,
                  "save synthesis time":None,
                  "contrastive_total_time":None,
                  "Origin_total_time":None,
                  "save total time":None,
                  }
    contrastive_dict = {"benchmark":benchmark,"type":"contrastive","start_synth":ini_synth}
    origin_dict = {"benchmark":benchmark,"type":"origin","start_synth":ini_synth}
    compare_dict = {"benchmark":"compare","type":"difference","start_synth":ini_synth}
    file_con = "results/contrastive/"+benchmark+"_parm_confidence_0.5_runs_50_radius_1_init_20_trainInterval_5.npy"
    file_origin = "results/origin/"+benchmark+"_parm_runs_50_radius_1_init_20.npy"
    if not pathlib.Path(file_con).exists() or not pathlib.Path(file_origin).exists():
        continue
    contrastive = np.load(file_con,allow_pickle=True).item()
    origin = np.load(file_origin,allow_pickle=True).item()
    if len(contrastive['timeEvolutions'])>50:
        print(benchmark)
    for i,item in enumerate(items):
        dc = contrastive[item]
        do = origin[item]
        contrastive_dict[csv_items[i]] = np.mean(dc)
        origin_dict[csv_items[i]] = np.mean(do)
        compare_dict[csv_items[i]] = origin_dict[csv_items[i]] - contrastive_dict[csv_items[i]]
        Total_dict["contrastive_"+csv_items[i]] = np.mean(dc)
        Total_dict["Origin_" + csv_items[i]] = np.mean(do)
    contrastive_dict["evolution_synth"] = contrastive_dict["mean_synth"] - ini_synth
    origin_dict["evolution_synth"] = origin_dict["mean_synth"] - ini_synth
    compare_dict["evolution_synth"] = origin_dict["evolution_synth"] - contrastive_dict["evolution_synth"]
    synthesis_time = mean_synthesize_time[benchmark]
    Total_dict["save synthesis time"] = synthesis_time*(origin_dict["mean_synth"] - contrastive_dict["mean_synth"])
    Total_dict["contrastive_total_time"] = Total_dict["contrastive_mean_time"] + synthesis_time*contrastive_dict["mean_synth"]
    Total_dict["Origin_total_time"] = Total_dict["Origin_mean_time"] + synthesis_time * origin_dict["mean_synth"]
    Total_dict["save total time"] = Total_dict["Origin_total_time"] - Total_dict["contrastive_total_time"]
    Total_dict["ADRS Difference"] = Total_dict["contrastive_mean_adrs"] - Total_dict["Origin_mean_adrs"]
    Total_dict["start"] = 20
    # (2) 添加行数据
    df = df.append(contrastive_dict, ignore_index=True)
    df = df.append(origin_dict, ignore_index=True)
    df = df.append(compare_dict, ignore_index=True)
    total_df = total_df.append(Total_dict, ignore_index=True)
df.to_csv("statistics/lattice.csv",index=False)
total_df.to_csv("statistics/total_lattice.csv",index=False)


