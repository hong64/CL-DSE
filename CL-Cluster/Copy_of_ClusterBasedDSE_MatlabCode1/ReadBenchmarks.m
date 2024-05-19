function [data, featureSets, discretizedFeatureSets] = ReadBenchmarks(benchmark,normalizeData)
    results = py.datasets.get_dse_description(benchmark);
    data = [];
    py_data = results{1};
    for i = 1:size(py_data,2)
        data = [double(py_data{i});data];
    end
    featureSets = cell(results{3});
    discretizedFeatureSets = cell(results{4});
    for i=1:size(featureSets,2)
        featureSets{i} = double(featureSets{i});
        discretizedFeatureSets{i} = double(discretizedFeatureSets{i});
    end
    ids = find(data(:,1)~=10000000);
    failed_ids = find(data(:,1) == 10000000);
    max_latency = max(data(ids,1));
    max_area = max(data(ids,2));
%     data(:,1) = data(:,1)/max(data(:,1));
%     data(:,2) = data(:,2)/max(data(:,2));
   data(:,1) = data(:,1)/max_latency;
   data(:,2) = data(:,2)/max_area;
   data(failed_ids,1) = 10000000;
   data(failed_ids,2) = 10000000;
end

