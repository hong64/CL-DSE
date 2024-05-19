function samples = extremeFeatureSampling(nOfSample, featureSets, a)
% This function samples nOfSample configurations according to a beta
% distribution. For each feature a specific value is sampled. It generates
% exactly nOfSample different configurations. The a value set the alpha
% value of the beta probability function. In our case alpha and beta are
% the same.

featuresSamples = [];
maxFeaturesValue = []; 
minFeatureValues = [];
for set=1:length(featureSets)
    step = 1/(length(featureSets{set}));
    x = 0:step:1;
    featuresCDFs{set} = betacdf(x,a,a);
    maxFeaturesValue = [maxFeaturesValue, max(featureSets{set})];
    minFeatureValues = [minFeatureValues, min(featureSets{set})];
end

featuresSamples(1,:) = maxFeaturesValue;
featuresSamples(2,:) = minFeatureValues;

for i=3:nOfSample
    for j=1:length(featureSets)
        fSet = featureSets{j};
        U = rand;
        sample(1,j) = fSet(sum(featuresCDFs{j}<=U));
    end
    while ismember(sample, featuresSamples, 'rows')
        for j=1:length(featureSets)
            fSet = featureSets{j};
            U = rand;
            sample(1,j) = fSet(sum(featuresCDFs{j}<=U));
        end
    end
    featuresSamples(i,:) = sample;
end

samples = featuresSamples;