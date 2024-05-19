function validConfig = castToValidConfiguration(newParetoFeatures, discretizedFeatureSets, featureSets, normalizeData, opt)
% Transforms the configuration generated during the exploration into
% synthesizable configurations. opt determines if we want to synthesize
% also the closer configuration or if we want to apply only upgrade and
% downgrade.
    config = [];
    if normalizeData
        downgradedSolutions = downgradeSolution(newParetoFeatures, discretizedFeatureSets);
    else
        downgradedSolutions = downgradeSolution(newParetoFeatures, featureSets);
    end
    uniqueDowngradedSolutions = downgradedSolutions;
    [uniqueDowngradedSolutions,~,~]=unique(uniqueDowngradedSolutions,'rows');
    if normalizeData
        upgradedSolutions = upgradeSolution(newParetoFeatures, discretizedFeatureSets);
    else
        upgradedSolutions = upgradeSolution(newParetoFeatures, featureSets);
    end
    uniqueUpgradedSolutions = upgradedSolutions;
    [uniqueUpgradedSolutions,~,~]=unique(uniqueUpgradedSolutions,'rows');

    config = [config; uniqueDowngradedSolutions; uniqueUpgradedSolutions];
    
    if opt <=2
        if normalizeData
            closerSolutions = closerSolution(newParetoFeatures, discretizedFeatureSets);
        else
            closerSolutions = closerSolution(newParetoFeatures, featureSets);
        end
        uniqueCloserSolutions = closerSolutions;
        [uniqueCloserSolutions,~,~]=unique(uniqueCloserSolutions,'rows');
        config = [config; uniqueCloserSolutions];
    end
    
    uniqueConfig = config;
    [uniqueConfig,~,~]=unique(uniqueConfig,'rows');
    validConfig = uniqueConfig;