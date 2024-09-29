classdef ClusterObj
    % Object created to define a cluster, it is defined by the points that
    % identify the cluster (Clusters). It provides function to calculate 
    % the centroid of the cluster and the related pareto points
    properties
        Cluster
        features
        paretoPointsIdx
        clusterVectorSums % Note: this is the sum centered to the origin, to obtain the real values the centroid has to be summed
    end
    methods
        function obj = ClusterObj(clust)
            obj.Cluster = clust;
            
        end
        function point = getCentroid(obj)
            c = zeros(1,size(obj.Cluster,2));
            for i=1:size(obj.Cluster,2)
                c(1,i) = sum(obj.Cluster(:,i))/length(obj.Cluster(:,i));
            end
            point = c;
        end
        function paretoIdx = findParetoIdx(obj)
            x = obj.Cluster(:,1);
            y = obj.Cluster(:,2);
            cl = obj.Cluster;
            if size(cl,1) == 1
                paretoIdx = [1];
            else
                if size(cl,1) == 2
                    paretoIdx = [1, 2];
                else
                    [~, paretoIdx] = paretoFrontier(cl);
                end
            end
        end
        function obj = set.clusterVectorSums(obj,vectorSums)
            obj.clusterVectorSums = vectorSums;
        end
    end
end