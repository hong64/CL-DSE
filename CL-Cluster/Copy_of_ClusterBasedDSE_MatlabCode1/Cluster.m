function [dataEvolution,sampledDataEvolution,startingPPEvolution,finalPPEvolution,timeEvolutions,onlineADRSEvolution,startingADRSEvolutions,synthEvolution] = Cluster(flag,nExp,data, sample, featureSets, discretizedFeatureSets, normalizeData, clustTradeOff,dataEvolution,sampledDataEvolution,startingPPEvolution,finalPPEvolution,timeEvolutions,onlineADRSEvolution,startingADRSEvolutions,synthEvolution)
    startTime = tic();
    if strcmp(flag, "Contrastive") == 1
        [startADRS, onlineADRS, startingPP, finalPP, synthNum, sampledData] = clusterbased_expl(data, sample, featureSets, discretizedFeatureSets, normalizeData, clustTradeOff);
    else
        [startADRS, onlineADRS, startingPP, finalPP, synthNum, sampledData] = clusterbased_expl_origin(data, sample, featureSets, discretizedFeatureSets, normalizeData, clustTradeOff);
    
    end
    execTime = toc(startTime);
    dataEvolution{nExp} = data;
    sampledDataEvolution{nExp} = sampledData;
    startingPPEvolution{nExp} = startingPP;
    finalPPEvolution{nExp} = finalPP;
    timeEvolutions{nExp} = transpose(execTime);
    onlineADRSEvolution{nExp} = onlineADRS;
    startingADRSEvolutions{nExp} = startADRS;
    synthEvolution{nExp} = transpose(synthNum);


end

