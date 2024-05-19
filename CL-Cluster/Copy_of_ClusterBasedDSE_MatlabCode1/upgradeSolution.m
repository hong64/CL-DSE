function featureSet = upgradeSolution(possibleFeatureSet, inputFeatureSets)

nSet = size(inputFeatureSets,2);
featureSet = zeros(nSet,1);
for i=1:size(possibleFeatureSet,1)
    for j=1:size(possibleFeatureSet,2)
        set = inputFeatureSets{j};
        tmp = 0;
        for k=1:length(set)            
            if set(k) >= possibleFeatureSet(i,j)
                tmp = set(k);
                break
            else
                tmp = set(k);
            end
        end
        featureSet(i,j) = tmp;
        tmp = 0;
    end
end