function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

loglikelihood = 0;
P.c = zeros(1,K);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
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
    for i = 1:size(G,1)
        if G(i,1) == 0
            [P.clg(i).mu_y(j),P.clg(i).sigma_y(j)] = FitGaussianParameters(ds(:,i,1));
            [P.clg(i).mu_x(j),P.clg(i).sigma_x(j)] = FitGaussianParameters(ds(:,i,2));
            [P.clg(i).mu_angle(j),P.clg(i).sigma_angle(j)] = FitGaussianParameters(ds(:,i,3));
        else
            parent_data = reshape(ds(:,G(i,2),:),[size(ds,1),size(dataset,3)]);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loglikelihood = ComputeLogLikelihood(P, G, dataset);
fprintf('log likelihood: %f\n', loglikelihood);

