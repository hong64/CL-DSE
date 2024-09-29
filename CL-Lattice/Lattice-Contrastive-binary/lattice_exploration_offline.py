
import sys
from Lattice_Contrastive import *
from Lattice_Origin import *
from datasets import get_dse_description
sys.setrecursionlimit(15200)
# sys.setrecursionlimit(10000)
np.random.seed(42)

# Set the radius for the exploration
radius = 1

# Set number of runs due to the probabilistic initial sampling
n_of_runs = 2

# To plot the ADRS chart
plot = False

# Initial sampling size
intial_sampling_size = 20

# Performance goal
pf_goal = 0.0000

# Collect stats
max_n_of_synth_C = 0
max_n_of_synth_O = 0
goal_stats_history = []
goal_stats = []

# Read dataset

# Specify the list of dataset to explore


success_benchmark =[
    # "ellpack_ellpack_spmv",
    # "bulk",
    # "md_kernel_knn_md",
    # "ncubed",
    # "bbgemm_blocked_gemm",
    # "fft_strided_fft",
    # "ms_mergesort_merge_sort",
    # 'stencil_stencil2d_stencil',
    # 'stencil3d_stencil3d_stencil',
    # "update_radix_sort",
    # "hist_radix_sort",
    # "init_radix_sort",
    # "sum_scan_radix_sort",
    # "last_step_scan_radix_sort",
    # "local_scan_radix_sort",
    # "ss_sort_radix_sort",
    # # 'aes_addRoundKey_aes_aes',
    # # 'aes_subBytes_aes_aes',
    'aes_addRoundKey_cpy_aes_aes',
    # 'aes_shiftRows_aes_aes',
    # # 'aes_mixColumns_aes_aes',
    # 'aes_expandEncKey_aes_aes',
    # 'aes256_encrypt_ecb_aes_aes',
    # 'get_oracle_activations1_backprop_backprop',
    # "get_oracle_activations2_backprop_backprop",
    # 'matrix_vector_product_with_bias_input_layer',
    # 'matrix_vector_product_with_bias_second_layer',
    # 'matrix_vector_product_with_bias_output_layer',
    # 'backprop_backprop_backprop',
    # 'add_bias_to_activations_backprop_backprop',
    # # 'soft_max_backprop_backprop',
    # 'take_difference_backprop_backprop',
    # 'update_weights_backprop_backprop',
    # 'get_delta_matrix_weights1',
    # 'get_delta_matrix_weights2',
    # 'get_delta_matrix_weights3',
    # "merge_merge_sort",
]



for k,b in enumerate(success_benchmark):
    print("\n---------------------------"+b+"--------------------------------------------\n")
    x, y, status, discretized_feature,original_feature = get_dse_description(b)
    intial_sampling_size = round(x.shape[0]*0.05)
    synthesis_result = y.tolist()
    configurations = x.tolist()
    entire_ds = []
    for i in range(0, len(synthesis_result)):
        entire_ds.append(DSpoint(synthesis_result[i][1], synthesis_result[i][0], list(configurations[i])))
    adrs_run_history_C = []
    run_stats_C = []

    adrs_run_history_O = []
    run_stats_O = []

    adrs_run_stats_C = []
    adrs_run_stats_O = []

    n_estimators = len(configurations)

    timeEvolutions_C = []
    timeEvolutions_O = []

    synthesize_evolutions_C = []
    synthesize_evolutions_O = []
    #保存每次运行的时间
    for run in range(0, n_of_runs):

        synthesize_ids_C,timeEvolutions_C,adrs_evolution_C,run_stats_C,samples,max_n_of_synth_C = Lattice_Contrastive(run,original_feature, radius,intial_sampling_size,entire_ds,timeEvolutions_C,max_n_of_synth_C)
        synthesize_ids_O,timeEvolutions_O, adrs_evolution_O, run_stats_O,max_n_of_synth_O = Lattice_Origin(run, original_feature, radius,
                                                                                       intial_sampling_size, entire_ds,
                                                                                       timeEvolutions_O, max_n_of_synth_O,samples)
        adrs_run_stats_C.append(run_stats_C)
        adrs_run_history_C.append(adrs_evolution_C)

        adrs_run_stats_O.append(run_stats_O)
        adrs_run_history_O.append(adrs_evolution_O)

        synthesize_evolutions_C.append(synthesize_ids_C)
        synthesize_evolutions_O.append(synthesize_ids_O)
        # goal_stats_history.append(goal_stats)
    save_Contrastive(adrs_run_stats_C, n_of_runs, max_n_of_synth_C, goal_stats, intial_sampling_size, timeEvolutions_C,b,radius,synthesize_evolutions_C)
    save_Origin(adrs_run_stats_O, n_of_runs, max_n_of_synth_O, goal_stats, intial_sampling_size, timeEvolutions_O,
                         b,radius,synthesize_evolutions_O)







