import copy

from ReadDB import  readDB
import numpy as np
import pandas as pd
from statistics_utils import calculate_EGA_mean_time,compare_EGA_pareto_points
benchmarks = [
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
def get_file_paths(benchmark,n_runs):
    file_con = "./EGA/contrastive/" + benchmark + "_runs_"+str(n_runs)+".npy"
    file_origin = "./EGA/origin/" + benchmark + "_runs_"+str(n_runs)+".npy"
    return file_con,file_origin

import pathlib
items = ["ADRSEvolutions",'SynthesisEvolutions',"timeEvolutions"]
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
                  "save total time",
                # "CL-EGA Paretos",
                # "EGA Paretos"
                                 ])
for m,benchmark in enumerate(benchmarks):
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
                  "contrastive_paretos": None,
                  "origin_paretos": None
                  }
    ini_synth = round(design_space_size[m] * 0.05)
    contrastive_dict = {"benchmark":benchmark,"type":"contrastive","start_synth":ini_synth}
    origin_dict = {"benchmark":benchmark,"type":"origin","start_synth":ini_synth}
    compare_dict = {"benchmark":"compare","type":"difference","start_synth":ini_synth}
    n_runs = 50
    file_con,file_origin = get_file_paths(benchmark,n_runs)
    if not pathlib.Path(file_con).exists() or not pathlib.Path(file_origin).exists():
        n_runs = 20
        file_con, file_origin = get_file_paths(benchmark, n_runs)
        if not pathlib.Path(file_con).exists() or not pathlib.Path(file_origin).exists():
            n_runs = 10
            file_con, file_origin = get_file_paths(benchmark, n_runs)
            if not pathlib.Path(file_con).exists() or not pathlib.Path(file_origin).exists():
                continue

    contrastive = np.load(file_con,allow_pickle=True).item()
    origin = np.load(file_origin,allow_pickle=True).item()
    for i,item in enumerate(items):
        dc = contrastive[item]
        do = origin[item]
        if item != 'SynthesisEvolutions':
            contrastive_dict[csv_items[i]] = np.mean(dc)
            origin_dict[csv_items[i]] = np.mean(do)
            compare_dict[csv_items[i]] = origin_dict[csv_items[i]] - contrastive_dict[csv_items[i]]
            Total_dict["contrastive_" + csv_items[i]] = np.mean(dc)
            Total_dict["Origin_" + csv_items[i]] = np.mean(do)
        else:
            contrastive_n_synths = [len(synthesis_designs) for synthesis_designs in dc]
            origin_n_synths = [len(synthesis_designs) for synthesis_designs in do]
            contrastive_dict[csv_items[i]] = np.mean(contrastive_n_synths)
            origin_dict[csv_items[i]] = np.mean(origin_n_synths)
            compare_dict[csv_items[i]] = origin_dict[csv_items[i]] - contrastive_dict[csv_items[i]]
            Total_dict["contrastive_" + csv_items[i]] = np.mean(contrastive_n_synths)
            Total_dict["Origin_" + csv_items[i]] = np.mean(origin_n_synths)

    contrastive_paretos = compare_EGA_pareto_points(contrastive['SynthesisEvolutions'],benchmark,n_runs)
    origin_paretos = compare_EGA_pareto_points(origin['SynthesisEvolutions'],benchmark,n_runs)
    contrastive_synthesis_time = calculate_EGA_mean_time(contrastive['SynthesisEvolutions'],benchmark,n_runs)
    origin_synthesis_time = calculate_EGA_mean_time(origin['SynthesisEvolutions'], benchmark, n_runs)
    contrastive_dict["evolution_synth"] = contrastive_dict["mean_synth"] - ini_synth
    origin_dict["evolution_synth"] = origin_dict["mean_synth"] - ini_synth
    compare_dict["evolution_synth"] = origin_dict["evolution_synth"] - contrastive_dict["evolution_synth"]
    Total_dict["save synthesis time"] = origin_synthesis_time-contrastive_synthesis_time
    Total_dict["contrastive_total_time"] = Total_dict["contrastive_mean_time"] + contrastive_synthesis_time
    Total_dict["Origin_total_time"] = Total_dict["Origin_mean_time"] + origin_synthesis_time
    Total_dict["save total time"] = Total_dict["Origin_total_time"] - Total_dict["contrastive_total_time"]
    Total_dict["ADRS Difference"] = Total_dict["contrastive_mean_adrs"] - Total_dict["Origin_mean_adrs"]
    Total_dict["start"] = ini_synth
    Total_dict["contrastive_paretos"] = contrastive_paretos
    Total_dict["origin_paretos"] = origin_paretos
    # (2) 添加行数据
    df = df.append(contrastive_dict, ignore_index=True)
    df = df.append(origin_dict, ignore_index=True)
    df = df.append(compare_dict, ignore_index=True)
    total_df = total_df.append(Total_dict, ignore_index=True)
df.to_csv("statistics/EGA.csv",index=False)
total_df.to_csv("statistics/total_EGA.csv",index=False)

    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # con_evolution = contrastive["adrs_mean"]
    # ori_evolution = origin["adrs_mean"]
    # con_synth = list(range(ini_synth,ini_synth+len(con_evolution)))
    # ori_synth = list(range(ini_synth, ini_synth +len(ori_evolution)))
    # plt.plot(con_synth,con_evolution)
    # plt.plot(ori_synth,ori_evolution, color='red', linewidth=1, linestyle='--')
    # plt.savefig("statistics/pits/"+benchmark+".png")
    # plt.close()


