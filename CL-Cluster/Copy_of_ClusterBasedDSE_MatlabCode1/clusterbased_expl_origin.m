function [startADRS, onlineADRS, startingPP, finalPP, synthNum, sampledData] = clusterbased_expl(data, sample, featureSets, discretizedFeatureSets, normalizeData, clustTradeOff)
clusterize =  true;
% totalSynth = size(sample,1);
% synthStory = totalSynth;
finalParetoPoint = [];
startParetoPoint = [];
% step = 0;
% steps = step;
sampledData = sample;
[paretoRealData, ~] = paretoFrontier(data);
[paretoSampleData, ~] = paretoFrontier(sample);
startParetoPoint = paretoSampleData;
startADRS = ADRS(paretoRealData,paretoSampleData);
% adrs = startADRS;
onlineADRS = startADRS;

%% continue to clusterize the synthesized data untill there are unsithesised configuration
while clusterize
%     step = step+1;
%     steps = [steps; step];
%     indexes =find(sampledData(:,1) == 20000000);
%     sampledData(indexes,:) = [];
    initialData = sampledData;
    
    %% Start clustering
    % Cluster sampled dataset
    d = pdist(sampledData,'seuclidean');
    link = linkage(d);
    
    % Apply clustering until the proper cluster tradoff is satisfied
    maxInconsistency = max(inconsistent(link));
    maxInconsistency = max(maxInconsistency);
    increment = 0.001;
    for i=0.1:increment:maxInconsistency
        c_sampled = cluster(link,'cutoff',i);
        clusterNumberSampled = max(c_sampled);
        if clusterNumberSampled <= size(sampledData,1)/clustTradeOff
            c_sampled = cluster(link,'cutoff',i-increment);
            clusterNumberSampled = max(c_sampled);
            break
        end
    end
    
    %% Extract clusters
    centroidsClusteredSample = zeros(clusterNumberSampled, size(sampledData,2));
    x = sampledData(:,1);
    y = sampledData(:,2);
    feature = sampledData(:,3:end);
    for i=1:clusterNumberSampled
        tmp = c_sampled == i;
        X = x(tmp);
        Y = y(tmp);
        Fs = feature(tmp,:);
        F = sum(Fs)/size(Fs,1);
        Cluster = [X,Y,Fs];
        clusters(i) = ClusterObj(Cluster);
        cent = getCentroid(clusters(i));
        centroidsClusteredSample(i,:) = cent;
        clusters(i).features = F;
    end
    
    %% Find pareto frontiers for the clusters
    % This step remove clusters that are dominated by other cluster
    % Calculate pareto cluster frontiers
    [paretoClustSampled_pts, paretoClustSampled_idx] = paretoFrontier(centroidsClusteredSample);
    
    %%  Apply clusters selection criterion other than pareto dominance
    closeFactor = 1;
    for i=1:length(clusters)
        % If already selected as pareto cluster skip it.
        if any(i == paretoClustSampled_idx)
            continue
        else
            % Check if centroid of cluster is inside a pareto cluster
            % boundary. If inside, include it to the set of pareto
            % clusters
            centroid = getCentroid(clusters(i));
            %% TODO: CHECK THIS CONDITION
            %                 if any((centroid(1,1) <= paretoSampleData(:,1)+max(paretoSampleData(:,1))*closeFactor & ...
            %                         centroid(1,2) <= paretoSampleData(:,2)+max(paretoSampleData(:,2))*closeFactor) == 1)
            %                         paretoClustSampled_idx = [paretoClustSampled_idx; i];
            %                         paretoClustSampled_pts = [paretoClustSampled_pts; centroid(1:2)];
            %                     continue
            %                 else
            if any((centroid(1,1) <= max(clusters(i).Cluster(:,1)) && centroid(1,1) >= min(clusters(i).Cluster(:,1))) && ...
                    (centroid(1,2) <= max(clusters(i).Cluster(:,2)) && centroid(1,2) >= min(clusters(i).Cluster(:,2))) == 1)
                paretoClustSampled_idx = [paretoClustSampled_idx; i];
                paretoClustSampled_pts = [paretoClustSampled_pts; centroid(1:2)];
                continue
            else
                % Check if a cluster contains solution that belong to
                % the pareto frontier of solutions. If that is the case
                % add the cluster to the set of pareto clusters.
                cl = clusters(i).Cluster;
                for j=1:size(cl,1)
                    if any(cl(j,1) <= paretoSampleData(:,1) & cl(j,2) <= paretoSampleData(:,2) == 1)
                        paretoClustSampled_idx = [paretoClustSampled_idx; i];
                        paretoClustSampled_pts = [paretoClustSampled_pts; centroid(1:2)];
                        break
                    end
                end
            end
        end
    end
    
    %% Start exploring clusters to find new solutions
    % Pick one cluster and explore possible solutions
    % Only dominating clusters are explored
    %sampledDataBeforeClustering = sample;
    for clust=1:size(paretoClustSampled_idx,1)
        cl = clusters(paretoClustSampled_idx(clust,1));
        
        % After intraclusterSums a new cluster exists with the new coordinates estimated form the sums of
        % the vectors
        newCluster = intraclusterExploration(cl);
        
        % Calculate the pareto frontier for the new cluster
        [~, paretoClustExtimated_idx] = paretoFrontier(newCluster);
        
        
        % Expand the data sampled with the new pareto point estimaded
        sampledData = [sampledData; newCluster(paretoClustExtimated_idx,:)];
        
        % Consider only uniqe solution. Some pareto points may be
        % added multiple times to the sampled data merging the new
        % solutions with the old.
        uniqueSampledData = sampledData;
        [uniqueSampledData,~,~]=unique(uniqueSampledData,'rows');
        
        % Note: uniqueSampledData contains the cordinates of the
        % original points (the firstly synthesised) and the new
        % estimated point to synthesise.
        % For the non synthesised points, area and latency
        % are only estimation. The configurations have to be
        % upgraded/downgraded before being synthesised.
    end
    
    %Once all the cluster have been expanded the global pareto frotier for the new estimated points is
    %calculated
    % Only the dominating point will be synthesised
    [~, paretoGlobalExtimated_idx] = paretoFrontier(uniqueSampledData);
    globalParetoPoints = uniqueSampledData(paretoGlobalExtimated_idx,:);
    % globalParetoPoints cointains only points I am interested to
    % synthesize. These poins are either new point and old points.
    newParetoFeatures = globalParetoPoints(:,3:end);
    
    % Cast the pareto solutions discovered into valid synthesizable
    % configurations
    intraclusterExplorationFeatures = castToValidConfiguration(newParetoFeatures, discretizedFeatureSets, featureSets, normalizeData, 2);
    
    % Now the new solution configurations only appears once
    % uniqueNewFeatures contains only configuration that are on the estimated pareto
    % curve (and may be on the real one), which I want to synthesise, but I don't want to resynth
    % already synthesiesd ones (the one synthesised in previous steps).
    AlreadySynthesized=ismember(intraclusterExplorationFeatures,initialData(:,3:end),'rows');

    toSynthesize = not(AlreadySynthesized);
    intraclusterExplorationFeatures = intraclusterExplorationFeatures(toSynthesize,:);
    
    %% Synthesis simulation
    % At this point I have identified the solution to synthesise.
    % This step retrive data from original synteshis. Synthesis simulation
    
    % The synthesis is performed element by element in order to get the
    % new ADRS at each new synthesis.
    newData = [];
    sampledData = initialData;
    for p=1:size(intraclusterExplorationFeatures,1)
        
        indx = ismember(data(:,3:end), intraclusterExplorationFeatures(p,:),'rows');

        newData = [newData; data(indx,1), data(indx,2), data(indx,3:end)];
    
        sampledData = [sampledData; newData];
            
        onlineADRS = [onlineADRS;ADRS(paretoRealData, paretoFrontier(sampledData))];
        
        
    end
    
    %% Get ready for inter-cluster exploration
    % If culster number greater than 1 perform inter cluster
    % exploration
    if clusterNumberSampled ~=1
        % if there are new data, update the clusters with the new
        % synthesised solutions
        if size(newData,1)~=0
            %Perform intercluster operations only for the data that have been
            %synthesised and that are on the pareto frontier
            [~, paretoSampledIdx] = paretoFrontier(sampledData);
            
            % Find new synthesised point in new pareto frontier
            indxNewSynth=ismember(sampledData(paretoSampledIdx,:), newData,'rows');
            % add the new data to the original clusters and then perform inter cluster analysis
            toClusterize = sampledData(indxNewSynth,:);
            % Update the old clusters with the new synthesied elements
            % adding each new element to the closer cluster
            for i=1:size(toClusterize,1)
                closerCluster = 0;
                closerCentroid = 0;
                tmpCluster = [];
                for j=1:length(paretoClustSampled_idx)
                    % Select cluster
                    cl = clusters(1,paretoClustSampled_idx(j));
                    % Select point
                    paretoP = toClusterize(i,:);
                    if j==1
                        closerCluster = j;
                        closerCentroid = cl.getCentroid;
                    else
                        centroid = cl.getCentroid;
                        %compute Euclidean distances:
                        dist = sqrt(sum(bsxfun(@minus, paretoP, centroid).^2,2));
                        if  dist < closerCentroid
                            closerCentroid = centroid;
                            closerCluster = j;
                        else
                            continue
                        end
                    end
                    tmpCluster = cl.Cluster;
                end
                tmpCluster = [tmpCluster; toClusterize(i,:)];
                clusters(paretoClustSampled_idx(closerCluster)).Cluster = tmpCluster;
            end
        end
        %% Inter-cluster exploration
        % perform intercluster exploration
        interclusterData = interclusterExploration(clusters, paretoClustSampled_idx, false);
        if size(interclusterData) ~= 0
            newEstimatedData = [globalParetoPoints; interclusterData];
        else
            newEstimatedData = globalParetoPoints;
        end
        
        % Now I have to find only the global pareto solution to synthesise
        % Only the dominating point will be synthesised
        
        % Now I want to find the pareto among the synthesised data, the new
        % one and the estimation given by the intercluster data.
        %%%%%% newEstimatedData = [globalParetoPoints; interclusterData];
    else
        newEstimatedData = globalParetoPoints;
    end
    [~, paretoGlobalInterclusteredExtimated_idx] = paretoFrontier(newEstimatedData);
    globalInterclusteredParetoPoints = newEstimatedData(paretoGlobalInterclusteredExtimated_idx,:);
    % globalParetoPoints cointains only points in which I am interested to
    % synthesize (also already synthesised point may be in that sat).
    % Those poins are either new point and old points.
    newInterclusteredParetoFeatures = globalInterclusteredParetoPoints(:,3:end);
    interclusterExplorationFeatures = castToValidConfiguration(newInterclusteredParetoFeatures, discretizedFeatureSets, featureSets, normalizeData, 2);
    AlreadySynthesized=ismember(interclusterExplorationFeatures,sampledData(:,3:end),'rows');
    toSynthesize = not(AlreadySynthesized);
    interclusterExplorationFeatures = interclusterExplorationFeatures(toSynthesize,:);
    
    
    %% Synthesis simulation
    % At this point I have identified the solution to synthesise.
    % This step retrive data from original synteshis. Synthesis simulation
    %         indx=ismember(data(:,3:dataCol), uniqueNewParetoFeatures,'rows');
    
    % The synthesis is performed element by element in order to get the
    % new ADRS at each new synthesis.
    newData2 = [];
    for p=1:size(interclusterExplorationFeatures,1)
        indxInterclustering=ismember(data(:,3:end), interclusterExplorationFeatures(p,:),'rows');
        
        newData2 = [newData2; data(indxInterclustering,1), data(indxInterclustering,2), data(indxInterclustering,3:end)];
        % Add the real synthesised data to the original
        % New data contains now the synthesised solutions
        sampledData = [sampledData; newData2];
        
        onlineADRS = [onlineADRS; ADRS(paretoRealData, paretoFrontier(sampledData))];
        
    end
    
    %% Stop condition
    if size(intraclusterExplorationFeatures,1) == 0 && size(interclusterExplorationFeatures,1) == 0
        clusterize = false;
    end
    
    % Calculate new pareto curves
    [paretoSampledData, ~] = paretoFrontier(sampledData);
    
    % Clear all the variables not necessary and close all the opened
    % figures
    close all;
    clearvars -except clusterize cutoffFactor data  dataCol initialData sample sampledData step totalSynth paretoSampledData paretoSampledIdx paretoInitialData paretoRealData paretoRealIdx paretoSampleData axis_limit finalData paretoClustSampled_pts featureSets discretizedFeatureSets normalizeData indxOpt explandedClusters featuresDistributions adrs finalParetoPoint startParetoPoint startADRS steps synthStory originalParetoSampledData clusterNumberSampledTotal clustTradeOff onlineADRS;
end

%% Final data
finalPP = paretoSampledData;
startingPP = startParetoPoint;
synthNum = size(onlineADRS,1)+size(sample,1);
end