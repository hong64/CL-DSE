function featureSet = downgradeSolution(possibleFeatureSet, inputFeatureSets)

nSet = size(inputFeatureSets,2);
featureSet = zeros(nSet,1);
for i=1:size(possibleFeatureSet,1)
    for j=1:size(possibleFeatureSet,2)
        set = inputFeatureSets{j};
        featSet = possibleFeatureSet(i,j);
        tmp = 0;
        for k=1:length(set)
            if set(k) <= featSet
                tmp = set(k);
            else
                if k == 1
                    tmp = set(k);
                end
                break
            end
        end
        featureSet(i,j) = tmp;
    end
end