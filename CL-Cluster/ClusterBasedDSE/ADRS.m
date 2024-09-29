function avgDistance = ADRS(referenceSet, approximateSet)
    nRefSetElements = size(referenceSet,1);
    nAppSetElements = size(approximateSet,1);
    sumOfMinDistances = 0;
    distances = zeros(nAppSetElements,1);
    for i=1:nRefSetElements
        for j=1:nAppSetElements
            distances(j) = dist(referenceSet(i,:), approximateSet(j,:));
        end
        sumOfMinDistances = sumOfMinDistances + min(distances);
    end
    
    avgDistance = sumOfMinDistances/nRefSetElements;
    
    function d = dist(refP, appP)
        x = (appP(1) - refP(1))/refP(1);
        y = (appP(2) - refP(2))/refP(2);
        toFindMax = [0,x,y];
        d = max(toFindMax);
      
    end
end