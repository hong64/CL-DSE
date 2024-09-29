from EGA_Origin import *
from datasets import get_dse_description
from EGA_Contrastive import *
import sys
from Assistant import TrainCondition
sys.setrecursionlimit(15200)
np.random.seed(42)

# Set the radius for the exploration
radius = 1

# Set number of runs due to the probabilistic initial sampling
n_of_runs = 10

# To plot the ADRS chart
plot = False


pf_goal = 0.0000

max_n_of_synth_C = 0
max_n_of_synth_O = 0
goal_stats_history = []
goal_stats = []


# Specify the list of dataset to explore
# benchmark = [
#     'add_bias_to_activations_backprop_backprop',
#     'aes_notable',
#     'aes_table',
#     'aes256_encrypt_ecb_aes_aes',
#     'aes_addRoundKey_aes_aes',
#     'aes_addRoundKey_cpy_aes_aes',
#     'aes_expandEncKey_aes_aes',
#     'aes_mixColumns_aes_aes',
#     'aes_shiftRows_aes_aes',
#     'aes_subBytes_aes_aes',
#     'backprop_backprop_backprop',
#     'bbgemm_blocked_gemm',
#     'bulk',
#     'ellpack_ellpack_spmv',
#     'fft_strided_fft',
#     'ncubed',
#     'get_delta_matrix_weights1',
#     'get_delta_matrix_weights2',
#     'get_delta_matrix_weights2_backprop_backprop',
#     'get_delta_matrix_weights3',
#     'get_oracle_activations1_backprop_backprop',
#     'get_oracle_activations2_backprop_backprop',
#     'hist_radix_sort',
#     'init_radix_sort',
#     'last_step_scan_radix_sort',
#     'local_scan_radix_sort',
#     'radix_local_scan',
#     'matrix_vector_product_with_bias_input_layer',
#     'matrix_vector_product_with_bias_output_layer',
#     'matrix_vector_product_with_bias_second_layer',
#     'md_kernel_knn_md',
#     'merge_merge_sort',
#     'ms_mergesort_merge_sort',
#     'soft_max_backprop_backprop',
#     'ss_sort_radix_sort',
#     'stencil2d',
#     'stencil_stencil2d_stencil',
#     'stencil3d_stencil3d_stencil',
#     'sum_scan_radix_sort',
#     'take_difference_backprop_backprop',
#     'twiddles8_transpose_fft',
#     'update_radix_sort',
#     'update_weights_backprop_backprop',
#     'viterbi_viterbi_viterbi']

success_benchmark = [
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
    # 'aes_mixColumns_aes_aes',
    # 'aes_expandEncKey_aes_aes',
    # 'aes256_encrypt_ecb_aes_aes',
    # 'ellpack_ellpack_spmv',
    # 'merge_merge_sort',
    # 'stencil3d_stencil3d_stencil',
    'aes_shiftRows_aes_aes',
    # 'get_oracle_activations1_backprop_backprop',
    # 'get_oracle_activations2_backprop_backprop',
    # 'matrix_vector_product_with_bias_input_layer',
    # 'matrix_vector_product_with_bias_output_layer',
    # 'matrix_vector_product_with_bias_second_layer',
    # 'backprop_backprop_backprop',
    # 'add_bias_to_activations_backprop_backprop',
    # 'soft_max_backprop_backprop',
    # 'take_difference_backprop_backprop',
    # 'update_weights_backprop_backprop',
    # 'stencil_stencil2d_stencil'
    # 'get_delta_matrix_weights1',
    # 'get_delta_matrix_weights2',
    # 'get_delta_matrix_weights3',
]


for k,b in enumerate(success_benchmark):
    # intial_sampling_size = round(design_space_size[k]*0.1)
    print("\n---------------------------"+b+"--------------------------------------------\n")
    x, y, status, discretized_feature,original_feature = get_dse_description(b)
    intial_sampling_size = round(x.shape[0]*0.05)
    if intial_sampling_size < 2:
        intial_sampling_size = 2
    synthesis_result = y.tolist()
    configurations = x.tolist()

    # Format data for the exploration
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
    synthesize_ids_C = []
    synthesize_ids_O = []

    discard_evolutions_C = []

    pareto_frontier_exhaustive, pareto_frontier_exhaustive_idx = GA_utils.pareto_frontier2d(entire_ds)
    for run in range(0, n_of_runs):
        train = TrainCondition(1,None)
        print("\n---------------------------Origin--------------------------------------------\n")
        origin_initial_ids = Origin(x, entire_ds, intial_sampling_size, original_feature, pareto_frontier_exhaustive, timeEvolutions_O,
                                    adrs_run_history_O, synthesize_evolutions_O, run)

        print("\n---------------------------Contrastive--------------------------------------------\n")
        Contrastive(train, discard_evolutions_C, x, entire_ds, intial_sampling_size,
                    original_feature, pareto_frontier_exhaustive, timeEvolutions_C, adrs_run_history_C,
                    synthesize_evolutions_C, run,origin_initial_ids)

    origin_results_dict = {"timeEvolutions":timeEvolutions_O,"ADRSEvolutions":adrs_run_history_O,"SynthesisEvolutions":synthesize_evolutions_O}
    np.save("EGA/origin/"+b+"_runs_"+str(n_of_runs)+".npy",origin_results_dict)

    contrastive_results_dict = {"timeEvolutions": timeEvolutions_C, "ADRSEvolutions": adrs_run_history_C,
                    "SynthesisEvolutions": synthesize_evolutions_C,"discard_evolutions_C":discard_evolutions_C}
    np.save("EGA/contrastive/" + b + "_runs_"+str(n_of_runs)+".npy", contrastive_results_dict)














