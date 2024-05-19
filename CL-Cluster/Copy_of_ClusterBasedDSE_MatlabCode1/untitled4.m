% 添加python的搜索路径
P = py.sys.path; 
if count(P,'D:\ClusterBasedDSE\Copy_of_ClusterBasedDSE_MatlabCode\Assistant.py') == 0
   insert(P,int32(0),'D:\ClusterBasedDSE\Copy_of_ClusterBasedDSE_MatlabCode\\Assistant.py');
end
