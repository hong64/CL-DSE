% 添加python的搜索路径
pyversion D:\anaconda\python.exe
P = py.sys.path; 
if count(P,'D:\paper code\ClusterBasedDSE\ClusterBasedDSE_MatlabCode\Assistant.py') == 0
   insert(P,int32(0),'D:\paper code\ClusterBasedDSE\ClusterBasedDSE_MatlabCode\Assistant.py');
end
py.importlib.import_module('Assistant')
% py.Assistant.Contrastive_Learning(1,1,1)
% pyenv(Version="D:\anaconda\python.exe")
% a = NaN;
% if isnan(a)
%     disp("yes")
% else
%     disp("No")
% end