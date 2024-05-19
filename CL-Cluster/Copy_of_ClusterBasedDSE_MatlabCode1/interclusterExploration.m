function interclusters = interclusterExploration(clusters, paretoClustSampled_idx, plot)
    % I want to calculate the intercluster exploration after the
    % intracluster exploration results have been performed synthesised
    interclusterUnion = zeros(size(paretoClustSampled_idx,1), size(paretoClustSampled_idx,1)); 
    interclusterUnion = [];
    for i=1:size(paretoClustSampled_idx,1)-1
        for j = 1:size(paretoClustSampled_idx,1)
            if i ~= j
                clI = clusters(i);
                clJ = clusters(j);
                ppClI = findParetoIdx(clI);
                ppClJ = findParetoIdx(clJ);
                clI.paretoPointsIdx = ppClI;
                clJ.paretoPointsIdx = ppClJ;
%                 paretoPntsClI = clI.Cluster(clI.paretoPointsIdx,:)
%                 paretoPntsClJ
                unionData = [clI.Cluster(clI.paretoPointsIdx,:); clJ.Cluster(clJ.paretoPointsIdx,:)];
                unionCluster = ClusterObj(unionData);
                unionCentroid = getCentroid(unionCluster);
                interClusterSumsData = intraclusterExploration(unionCluster);
                interclusterUnion = [interclusterUnion; intraclusterExploration(unionCluster)];
                if plot
                    figure
                    scatter(unionCentroid(:,1),unionCentroid(:,2),'*','red');
                    hold on;
                    grid on;

                    scatter(interClusterSumsData(:,1),interClusterSumsData(:,2),'o','green');
                    scatter(unionData(:,1),unionData(:,2),'o','blue');
                end
            end
        end
    end
    
    interclusters = interclusterUnion;
    
% %     interclusterUnion = [interclusterUnion; uniqueSampledData];
%     
% %     uniqueInterClusterData = interclusterUnion;
% %     [uniqueInterClusterData,~,~]=unique(uniqueInterClusterData,'rows');
%         
% %     [paretoInterClustExtimated_pts, paretoInterClustExtimated_idx] = paretoFrontier(interclusterUnion);
% 
%     [paretoClustExtimated_pts, paretoClustExtimated_idx] = paretoFrontier(newCluster);
%     % Expand the data sampled with the new intercluster pareto point estimaded
%     %sampledData = [sampledData; uniqueInterClusterData(paretoInterClustExtimated_idx,:)];
%     sampledData = [sampledData; uniqueSampledData(paretoClustExtimated_idx,:)];
%     %sampledData = [sampledData; newCluster(paretoInterClustExtimated_idx,:)];
%     % Consider only uniqe solution; some pareto points may be selected
%     % added multiple times to the sampled data merging the new solutions with the old.
%     uniqueSampledData = sampledData;
%     [uniqueSampledData,~,~]=unique(uniqueSampledData,'rows');