% pyversion D:\anaconda\python.exe
% P = py.sys.path; 
% if count(P,'D:\paper code\ClusterBasedDSE\ClusterBasedDSE_MatlabCode\datasets.py') == 0
%    insert(P,int32(0),'D:\paper code\ClusterBasedDSE\ClusterBasedDSE_MatlabCode\datasets.py');
% end
% py.importlib.import_module('datasets')
name = "aes_table";
results = py.datasets.get_dse_description(name);
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