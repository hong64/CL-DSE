clear all;
prefix = "./explorations_results/";
benchNames = {
% 'add_bias_to_activations_backprop_backprop';
% 'aes_expandEncKey_aes_aes';
% 'aes_table';
% 'aes256_encrypt_ecb_aes_aes';
% 'backprop_backprop_backprop';
% 'ellpack_ellpack_spmv';
% 'get_oracle_activations2_backprop_backprop';
% 'matrix_vector_product_with_bias_input_layer';
% 'matrix_vector_product_with_bias_output_layer';
% 'matrix_vector_product_with_bias_second_layer';
%     'merge_merge_sort';
%     'take_difference_backprop_backprop';
%     'update_weights_backprop_backprop';
%     'aes_notable';
    'aes_addRoundKey_aes_aes';
'aes_addRoundKey_cpy_aes_aes';
};
design_space_size = {
    1372;
    216;
    3072;
    1944;
    2048;
    1600;
    1372;
    1372;
    392;
    686;
    4096;
    512;
    1024;
    12288;
    500;
    625;
};
for i = 1:size(benchNames,1)
    contrastive_results = load(prefix+"contrastive/timed_"+benchNames(i)+"_expN50_prob_iS15_cF15");
%     prefix_o = load(prefix+"origin/timed_"+benchNames(i)+"_expN50_prob_iS15_cF15");
    syn1 = mean(cell2mat(synthEvolution));
    time1 = mean(timeEvolutions);
    adrs1 = mean(finalADRS);
    start = round(design_space_size{i}*percentage/100);
    origin_results = load(prefix+"origin/timed_"+benchNames(i)+"_expN50_prob_iS15_cF15");
    syn2 = mean(cell2mat(synthEvolution));
    time2 = mean(timeEvolutions);
    adrs2 = mean(finalADRS);



