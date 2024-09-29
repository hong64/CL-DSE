
function newCluster=intraclusterExploration(cl)
% intraclusterSums receives a cluster as input and return a vector with all the intracluster vector sums
    % Compute pareto frontier for each clusters
    pIdx = findParetoIdx(cl);
    cl.paretoPointsIdx = (pIdx);
    % Move centrod of the cluster into the origin
    clToOrig = cl.Cluster - getCentroid(cl);
    % Calculate vectors sum between pareto solutons
    clToOrigPareto = clToOrig(cl.paretoPointsIdx,:);
    clSums = zeros(size(clToOrigPareto,1), size(clToOrigPareto,1),size(clToOrigPareto,2));
    for i=1:size(clToOrigPareto,1)
        for j=1:size(clToOrigPareto,1)
            for z=1:size(clToOrigPareto,2)
                if i == j
                    clSums(i,j,z) = 0;
                else
                    clSums(i,j,z) = clToOrigPareto(i,z) + clToOrigPareto(j,z);
                end
            end
        end    
    end
    
    clSumsVectors = [];
    for i=1:size(clToOrigPareto,1)-1
        for j=i+1:size(clToOrigPareto,1)
            clSumsVectors = [clSumsVectors squeeze(clSums(i,j,:))];
        end
    end

    % Store vector sums pair into the object
    clSumsVectorsTrans = transpose(clSumsVectors);
    cl.clusterVectorSums = clSumsVectorsTrans;
    
     newCluster = [clSumsVectorsTrans; clToOrig] + getCentroid(cl);