function discretizedFeatSets = discretizeSet(featureSets)
    discretizedFeatureSets = {};
    featureSetsSizes = [];
    for set=1:length(featureSets)
        featureSetsSizes(set) = length(featureSets{set});
        step = 1/(featureSetsSizes(set)-1);
        discretizedFeature = [];
        for i=0:length(featureSets{set})-1
            discretizedFeature = [discretizedFeature, step*i];
        end
        discretizedFeatureSets{set} = discretizedFeature;
    end
    discretizedFeatSets = discretizedFeatureSets;