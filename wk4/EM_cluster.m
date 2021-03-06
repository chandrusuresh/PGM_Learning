% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(i,:,:) yields the 10x3 matrix for pose i.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(i,j) is the probability that example i belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to class j

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);

ClassProb = InitialClassProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  %
  % Fill in P.c with the estimates for prior class probabilities
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  %
  % Hint: This part should be similar to your work from PA8
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for k = 1:K
    P.c(k) = mean(ClassProb(:,k));
    for i = 1:size(G,1)
      if G(i,1) == 0
          [P.clg(i).mu_y(k),P.clg(i).sigma_y(k)] = FitG(poseData(:,i,1),ClassProb(:,k));
          [P.clg(i).mu_x(k),P.clg(i).sigma_x(k)] = FitG(poseData(:,i,2),ClassProb(:,k));
          [P.clg(i).mu_angle(k),P.clg(i).sigma_angle(k)] = FitG(poseData(:,i,3),ClassProb(:,k));
      else
          parent_data = reshape(poseData(:,G(i,2),:),[size(poseData,1),size(poseData,3)]);
          class_data = reshape(poseData(:,i,:),[size(poseData,1),size(poseData,3)]);
          [theta1,P.clg(i).sigma_y(k)] = FitLG(class_data(:,1),parent_data,ClassProb(:,k));
          [theta2,P.clg(i).sigma_x(k)] = FitLG(class_data(:,2),parent_data,ClassProb(:,k));
          [theta3,P.clg(i).sigma_angle(k)] = FitLG(class_data(:,3),parent_data,ClassProb(:,k));
          P.clg(i).theta(k,1:4) = [theta1(4),theta1(1:3)'];
          P.clg(i).theta(k,5:8) = [theta2(4),theta2(1:3)'];
          P.clg(i).theta(k,9:12) = [theta3(4),theta3(1:3)'];
      end
    end
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % E-STEP to re-estimate ClassProb using the new parameters
  %
  % Update ClassProb with the new conditional class probabilities.
  % Recall that ClassProb(i,j) is the probability that example i belongs to
  % class j.
  %
  % You should compute everything in log space, and only convert to
  % probability space at the end.
  %
  % Tip: To make things faster, try to reduce the number of calls to
  % lognormpdf, and inline the function (i.e., copy the lognormpdf code
  % into this file)
  %
  % Hint: You should use the logsumexp() function here to do
  % probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  currDir = cd;
  altDir = '../wk3/';
  prob = zeros(size(ClassProb));
  TotalProb = zeros(size(poseData,1),1);
  for i = 1:size(poseData,1)
      cd(altDir);
      for class = 1:K
        prob(i,class) = log(P.c(class)) + ComputeProbability_Obs_given_class(P,G,poseData(i,:,:),class);
      end
      cd(currDir);
      TotalProb(i) = logsumexp(prob(i,:));
  end
  ClassProb = exp(prob - TotalProb);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Compute log likelihood of dataset for this iteration
  % Hint: You should use the logsumexp() function here
  loglikelihood(iter) = 0.0;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  cd(altDir);
  loglikelihood(iter) = ComputeLogLikelihood(P, G, poseData);
  cd(currDir);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting: when loglikelihood decreases
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
