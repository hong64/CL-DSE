% function featureSet = closerSolution(possibleFeatureSet, varargin)
function featureSet = closerSolution(possibleFeatureSet, inputFeatureSets)

nSet = size(inputFeatureSets,2);
featureSet = zeros(1,nSet);
for i=1:size(possibleFeatureSet,1)
    for j=1:size(possibleFeatureSet,2)
        set = inputFeatureSets{j};
        tmp = 0;
        tmpVec = [];
        possibleFeature = possibleFeatureSet(i,j);
        for k=1:length(set)
            setK = set(k);
            if setK <= possibleFeature
                if k == length(set)
                    tmpVec = [setK];
                    break
                else
                    downgrade = setK;
                    upgrade = set(k+1);
                    tmpVec = [downgrade, upgrade];
                end
            else
                if k == 1
                    tmpVec = [setK];
                end
                break
            end    
        end
        [~,idx] = min(abs(tmpVec-possibleFeature));
        tmp = tmpVec(idx);
        featureSet(i,j) = tmp;
        tmp = 0;
    end
end