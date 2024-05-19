clear;
close all;

%% Add path to the databases folder
addpath('../ClusterBasedDSE_Datasets')

%% Exploration and visualization options
% Normalize dimensions
normalizeData = true;

% Specify which benchmark we want to explore
% A list containing the indices in order to refer to the benchNames
% entries.
% We used a list in order to run explorations over different benchmarks.
% The benchmarksList should contain only the indices of the benchmark for
% which we want to perform the exploration.
benchmarksList = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27];
% benchmarksList = [4];
% benchNames = {'ChenIDCt';'Autocorrelation';'adpcmEncode';'adpcmDecode';'reflectionCoeff';...
%     'aes256_encrypt_ecb';'ellpack';'matrix';'merge';'aes_notable'};
benchNames = {
%     'bulk'
%     'md_kernel_knn_md';
%     'viterbi_viterbi_viterbi';
%     'ncubed';
%     'bbgemm_blocked_gemm';
%     'fft_strided_fft';
%     'twiddles8_transpose_fft';
%     'ms_mergesort_merge_sort';
%     'update_radix_sort';
%     'hist_radix_sort';
%     'init_radix_sort';
%     'sum_scan_radix_sort';
%     'last_step_scan_radix_sort';
%     'local_scan_radix_sort';
%     'ss_sort_radix_sort';
%     
% 'aes_addRoundKey_aes_aes';
%     'aes_subBytes_aes_aes';
% 'aes_addRoundKey_cpy_aes_aes';
% 'aes_shiftRows_aes_aes';
%    'aes_mixColumns_aes_aes';
    'aes_expandEncKey_aes_aes';
%     'aes256_encrypt_ecb_aes_aes';
%     'stencil3d_stencil3d_stencil';
%     'merge_merge_sort',
%     'ellpack_ellpack_spmv'
%     'get_oracle_activations1_backprop_backprop';
%     'get_oracle_activations2_backprop_backprop';
% 'matrix_vector_product_with_bias_input_layer';
%     'matrix_vector_product_with_bias_output_layer';
%     'matrix_vector_product_with_bias_second_layer';
% 'backprop_backprop_backprop';
%     'add_bias_to_activations_backprop_backprop';
%     'soft_max_backprop_backprop';
%     'take_difference_backprop_backprop';
%     'update_weights_backprop_backprop';
% 'stencil_stencil2d_stencil';
% 'get_delta_matrix_weights1';
% 'get_delta_matrix_weights2';
% 'get_delta_matrix_weights3';
};
% Specify number of exploration to run (final results are averaged)
nOfExplorations = 20;

% Specify initial sampling techniques, if 0 it uses a probabilistic
% (beta distribution) sampling, 1 for a random sampling
samplingStrategy = 0;

% Specify how many sample to perform before the exploration (in percentage
% respect to the total number of configuration). Put the sizes inside a
% list. This, in order to perfomr benchamrking changing the initial 
% sampling size
% E.g. to perform exploration for different percentage: 
% initialSamplingPercentage = [2.5, 5, 10, 15, 20]; 
initialSamplingPercentage = [5];

% Specify a tradeoff between number of cluster to evaluate and how much to
% prune. # of cluster limited by # of point/clustTradeOff. 
% Lower trade-off = More clusters -> more pruning, less precision
% Higher trade-off = Less clusters -> Less pruning, more precision
% Requires a list stracture in order to perform benchmarking.
clustTradeOffList = [5];
% clustTradeOffList = [5,10,15,20];

% Generates final plot with curves
visualize_plots = true;

% Storing output data in a folder ./explorations_results.
% If the folder doesn't exists this will be created
saveData = true;

%% Exploration over different configurations
for b=1:length(benchmarksList)
    for p=1:length(initialSamplingPercentage)
        for cTO=1:length(clustTradeOffList)
            
            benchmarks = benchmarksList(b);
            percentage = initialSamplingPercentage(p);
            clustTradeOff = clustTradeOffList(cTO);

%% Single exploration            
%% Output dat file name
datFileName = benchNames(benchmarks,1);
datFileName = strcat('timed_',datFileName);
datFileName = strcat(datFileName,'_expN',num2str(nOfExplorations));
if samplingStrategy == 1
    datFileName = strcat(datFileName,'_ted');
else
    if samplingStrategy == 0
    	datFileName = strcat(datFileName,'_prob');
    else
	datFileName = strcat( datFileName,'_rand');
    end
end
datFileName = strcat(datFileName,'_iS',num2str(percentage));
datFileName = strcat(datFileName,'_cF',num2str(clustTradeOff));
toPrint = strcat('Start with the creation of\t',datFileName{1},'\n');
fprintf(toPrint);
[data, featureSets, discretizedFeatureSets] = ReadBenchmarks(benchNames{benchmarks},normalizeData);
dataEvolution = {};
sampledDataEvolution = {};
startingPPEvolution = {};
finalPPEvolution = {};
timeEvolutions = {};
onlineADRSEvolution = {};
startingADRSEvolutions = {};
synthEvolution = {};

dataEvolution_Origin = {};
sampledDataEvolution_Origin = {};
startingPPEvolution_Origin = {};
finalPPEvolution_Origin = {};
timeEvolutions_Origin = {};
onlineADRSEvolution_Origin = {};
startingADRSEvolutions_Origin = {};
synthEvolution_Origin = {};
%% Sample data according to a random choice
for nExp=1:nOfExplorations
    nOfSamples = round(length(data)*percentage/100);
%     nOfSamples = 5;
    if samplingStrategy == 0
        featureSample = extremeFeatureSampling(nOfSamples,discretizedFeatureSets, 0.2);
        indx=ismember(data(:,3:end), featureSample,'rows');
        sample = data(indx,:);
    else
        selected_data = datasample(data(:,3:end),nOfSamples,'Replace',false);
        indx=ismember(data(:,3:end), selected_data,'rows');
        sample = data(indx,:);
    end


    
    [dataEvolution,sampledDataEvolution,startingPPEvolution,finalPPEvolution,timeEvolutions,onlineADRSEvolution,startingADRSEvolutions,synthEvolution] = Cluster("Contrastive",nExp,data, sample, featureSets, discretizedFeatureSets, normalizeData, clustTradeOff,dataEvolution,sampledDataEvolution,startingPPEvolution,finalPPEvolution,timeEvolutions,onlineADRSEvolution,startingADRSEvolutions,synthEvolution);
    [dataEvolution_Origin,sampledDataEvolution_Origin,startingPPEvolution_Origin,finalPPEvolution_Origin,timeEvolutions_Origin,onlineADRSEvolution_Origin,startingADRSEvolutions_Origin,synthEvolution_Origin] = Cluster("Origin",nExp,data, sample, featureSets, discretizedFeatureSets, normalizeData, clustTradeOff,dataEvolution_Origin,sampledDataEvolution_Origin,startingPPEvolution_Origin,finalPPEvolution_Origin,timeEvolutions_Origin,onlineADRSEvolution_Origin,startingADRSEvolutions_Origin,synthEvolution_Origin);
    disp(nExp);

end

save_results(data,percentage,benchmarksList,nOfExplorations,datFileName,visualize_plots,saveData,benchNames,dataEvolution,sampledDataEvolution,startingPPEvolution,finalPPEvolution,timeEvolutions,onlineADRSEvolution,startingADRSEvolutions,synthEvolution,"contrastive")
save_results(data,percentage,benchmarksList,nOfExplorations,datFileName,visualize_plots,saveData,benchNames,dataEvolution_Origin,sampledDataEvolution_Origin,startingPPEvolution_Origin,finalPPEvolution_Origin,timeEvolutions_Origin,onlineADRSEvolution_Origin,startingADRSEvolutions_Origin,synthEvolution_Origin,"origin")



        end
    end
end
