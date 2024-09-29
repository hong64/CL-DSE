function [] = save_results(data,percentage,benchmarksList,nOfExplorations,datFileName,visualize_plots,saveData,benchNames,dataEvolution,sampledDataEvolution,startingPPEvolution,finalPPEvolution,timeEvolutions,onlineADRSEvolution,startingADRSEvolutions,synthEvolution,type)
    if visualize_plots
        % Boxplot containing info about the explorations performed
        figure
        subplot(1,3,1)
        finalADRS = [];
        for a=1:size(onlineADRSEvolution,2)
            finalADRS = [finalADRS; onlineADRSEvolution{a}(end)];
        end
        boxplot(finalADRS)
        grid on
%         ylim([0 max([startingADRSEvolutions{:}])+max([startingADRSEvolutions{:}]/10)])
        title('ADRS after DSE')
    
        subplot(1,3,2)
        boxplot([startingADRSEvolutions{:}])
        grid on
%         ylim([0 max([startingADRSEvolutions{:}])+max([startingADRSEvolutions{:}]/10)])
        title('ADRS before DSE')
    
        subplot(1,3,3)
        boxplot([synthEvolution{:}])
        grid on
        title('# of synthesis')
    
        % Plot ADRS evolution with respect to # of synthesis, the curve ends at 40% of DS size
        % that doesn't mean that 40% of design space has been synthesised
        onlineADRS = [];
        for i=1:length(onlineADRSEvolution)
            adrsTmp = onlineADRSEvolution{i};
            tmp = padarray(adrsTmp,length(data),adrsTmp(length(adrsTmp)),'post');
            onlineADRS = [onlineADRS tmp(1:round(length(data)*40/100),1)];
        end
        meanOnlineADRS = [];
        for i=1:size(onlineADRS,1)
            meanOnlineADRS = [meanOnlineADRS; mean(onlineADRS(i,:))];
        end
        x = linspace(round(length(data)*percentage/100),round(length(data)*percentage/100)+round(length(data)*40/100),round(length(data)*40/100));
        figure
        hold on
        grid on
        plot(x,meanOnlineADRS,'linewidth',1)
        xlabel('# of synthesis')
        ylabel('mean ADRS')
        
        txtData = [x',meanOnlineADRS];
        fid = fopen(['result\' benchNames{benchmarksList(1)} '.txt'],'wt');
        [m,n] = size(txtData);
        for i = 1 : m
            for j = 1 : n
                fprintf(fid,'%d\t',txtData(i,j));
            end
            fprintf(fid,'\n');
        end
    
        if nOfExplorations==1
        % Plot exhaustive exploration with respect to synthesis performed and
        % pareto frontier discovered
            figure
            hold on
            grid on
            scatter(dataEvolution{1}(:,1), dataEvolution{1}(:,2),'xr')
            scatter(sampledDataEvolution{1}(:,1), sampledDataEvolution{1}(:,2),'oc', 'filled')
            scatter(finalPPEvolution{1}(:,1), finalPPEvolution{1}(:,2),'ob', 'filled')
            xlabel('latency')
            ylabel('area')
        end
    end
    
    if saveData
        if exist('./explorations_results','dir') == 0
           mkdir('./explorations_results')
        end
        char(strcat('./',datFileName,'.mat'));
        if strcmp(type,"contrastive")
            save(char(strcat('./explorations_results/contrastive/',datFileName,'.mat')));
        else
            save(char(strcat('./explorations_results/origin/',datFileName,'.mat')));
        end
    end
    
end

