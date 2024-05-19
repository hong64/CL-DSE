function [pareto_pts, pareto_idx]  = paretoFrontier(data)
% It calculates the pareto frontier for a 2 column vector of data.
% Generates as output the set of pareto points and the indices
% corresponding to the pareto solution with respect to data.
    if size(data,1) == 1
        pareto_pts = [data(1,1) data(1,2)];
        pareto_idx = [1, 1];
    else
        if size(data,1) == 2
            px = [data(1,1); data(2,1)];
            py = [data(1,2); data(2,2)];
            pareto_pts = [px py];
            pareto_idx = [1, 2];
        else
            dataSorted = sortrows(data(:,1:2));
            paretoP = dataSorted(1,:);
            idx = 1;
            for i=2:size(dataSorted,1)
                if dataSorted(i,2) < paretoP(idx,2)
                    idx = idx + 1;
                    paretoP = [paretoP; dataSorted(i,:)];
                end
        
            end
            pareto_pts = paretoP;
            [~, indxP] = ismember(paretoP, data(:,1:2),'rows');
            pareto_idx = indxP(:,1);
        end
    end