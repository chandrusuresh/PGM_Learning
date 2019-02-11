function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:size(dataset,1)
    prob = 0.0;
    for j = 1:length(P.c)
        pc = 0.0;
        for k = 1:length(P.clg)
            if G(k,1) == 0
                pc = pc + lognormpdf(dataset(i,k,1),P.clg(k).mu_y(j),P.clg(k).sigma_y(j)) + ...
                        + lognormpdf(dataset(i,k,2),P.clg(k).mu_x(j),P.clg(k).sigma_x(j)) + ...
                        + lognormpdf(dataset(i,k,3),P.clg(k).mu_angle(j),P.clg(k).sigma_angle(j));
            else
                parent_val = dataset(i,G(k,2),:);
                data_val   = dataset(i,k,:);
                mu_y = sum(P.clg(k).theta(j,1:4).*[1,parent_val(:)']);
                mu_x = sum(P.clg(k).theta(j,5:8).*[1,parent_val(:)']);
                mu_angle = sum(P.clg(k).theta(j,9:12).*[1,parent_val(:)']);
                pc = pc + lognormpdf(data_val(1,1,1),mu_y,P.clg(k).sigma_y(j)) + ...
                          lognormpdf(data_val(1,1,2),mu_x,P.clg(k).sigma_x(j)) + ...
                          lognormpdf(data_val(1,1,3),mu_angle,P.clg(k).sigma_angle(j));
            end
        end     
        prob = prob + exp(log(P.c(j)) + pc);
    end
    loglikelihood = loglikelihood + log(prob);
end
