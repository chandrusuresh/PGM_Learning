function [P G loglikelihood] = LearnGraphAndCPDs(dataset, labels)

% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha) 
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

G = zeros(10,2,K); % graph structures to learn
% initialization
for k=1:K
    G(2:end,:,k) = ones(9,2);
end

% estimate graph structure for each class
for k=1:K
    % fill in G(:,:,k)
    % use ConvertAtoG to convert a maximum spanning tree to a graph G
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % YOUR CODE HERE
    %%%%%%%%%%%%%%%%%%%%%%%%%
    ind = find(labels(:,k) == 1);
    [A W] = LearnGraphStructure(dataset(ind,:,:));
    G(:,:,k) = ConvertAtoG(A);
end

% estimate parameters

P.c = zeros(1,K);
% compute P.c
% the following code can be copied from LearnCPDsGivenGraph.m
% with little or no modification
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE

def_struct = struct('mu_y',[],'sigma_y',[],...
        'mu_x',[],'sigma_x',[],...
        'mu_angle',[],'sigma_angle',[],...
        'theta',[]);
P.clg = [def_struct];
for i = 1:size(G,1)
    P.clg(i) = def_struct;
end
for j = 1:K
    ind = find(labels(:,j) == 1);
    P.c(j) = length(ind)/size(labels,1);
    ds = dataset(ind,:,:);
    for i = 1:size(G(:,:,j),1)
        if G(i,1,j) == 0
            [P.clg(i).mu_y(j),P.clg(i).sigma_y(j)] = FitGaussianParameters(ds(:,i,1));
            [P.clg(i).mu_x(j),P.clg(i).sigma_x(j)] = FitGaussianParameters(ds(:,i,2));
            [P.clg(i).mu_angle(j),P.clg(i).sigma_angle(j)] = FitGaussianParameters(ds(:,i,3));
        else
            parent_data = reshape(ds(:,G(i,2,j),:),[size(ds,1),size(dataset,3)]);
            class_data = reshape(ds(:,i,:),[size(ds,1),size(dataset,3)]);
            [theta1,P.clg(i).sigma_y(j)] = FitLinearGaussianParameters(class_data(:,1),parent_data);
            [theta2,P.clg(i).sigma_x(j)] = FitLinearGaussianParameters(class_data(:,2),parent_data);
            [theta3,P.clg(i).sigma_angle(j)] = FitLinearGaussianParameters(class_data(:,3),parent_data);
            P.clg(i).theta(j,1:4) = [theta1(4),theta1(1:3)'];
            P.clg(i).theta(j,5:8) = [theta2(4),theta2(1:3)'];
            P.clg(i).theta(j,9:12) = [theta3(4),theta3(1:3)'];
        end
    end
end

loglikelihood = ComputeLogLikelihood(P, G, dataset);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('log likelihood: %f\n', loglikelihood);