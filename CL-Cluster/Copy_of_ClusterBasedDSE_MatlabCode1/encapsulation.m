function [promising_flag,trainCondition] = encapsulation(sampledData,new_configuration,pareto_frontier,is_promising,trainCondition)
    
    history = py.list();
    P = py.list();
    for i=1:size(sampledData,1)
        if (sampledData(i,1) == 10000000 || sampledData(i,2) == 10000000) || (sampledData(i,1) == 20000000 || sampledData(i,2) == 20000000)
            continue
        end
        latency = sampledData(i,1);
        area = sampledData(i,2);
        history.append(py.Assistant.DSpoint(latency,area,sampledData(i,3:end)))
    end
    for i=1:size(pareto_frontier,1)
        latency_p = pareto_frontier(i,1);
        area_p = pareto_frontier(i,2);
        P.append(py.Assistant.DSpoint(area_p,latency_p,pareto_frontier(i,3:end)))
    end
    new_configuration = py.numpy.array(new_configuration).tolist();
    R = py.Assistant.Contrastive_Learning(history,new_configuration,P,trainCondition);
    promising_flag = R{1};
    trainCondition = R{2};
end
