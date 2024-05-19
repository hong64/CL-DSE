function discretizedDataset = discretizeFeature(featureSets, discretizedFeatureSets, data)
% This function discretize the feature sets. Each feature will be defined
% as a vector of elements equally distantiated in a range from 0 to 1.
    for i = 1:length(discretizedFeatureSets)
        discretizedSet = discretizedFeatureSets{i};
        featureSet = featureSets{i};
        for j=1:length(discretizedSet)
            tmp = data(:,i) == featureSet(j);
            idx=tmp==1;
            data(idx,i) = discretizedSet(j);
        end
    end
    discretizedDataset =  data;


