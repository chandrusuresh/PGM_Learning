clear all; close all; clc;
load PA9Data.mat;
currDir = cd;
c = 1;
cd ../wk3
ds = [];
for i = 1:length(datasetTrain3)
    for j = 1:size(datasetTrain3(i).actionData,2)
        ds1 = datasetTrain3(i).poseData(datasetTrain3(i).actionData(j).marg_ind,:,:);
        ds = [ds;ds1];
    end
end

for obj = 1:10
    figure(obj); hold on;
    sc = scatter3(ds(:,obj,1),ds(:,obj,2),ds(:,obj,3)*180/pi,'MarkerEdgeColor','b','MarkerFaceColor','b');
%     sc = scatter(ds(:,obj,1),ds(:,obj,2),'MarkerEdgeColor','b','MarkerFaceColor','b');
    sc.MarkerEdgeAlpha=0.3;
    sc.MarkerFaceAlpha=0.3;
end