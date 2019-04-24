% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P, loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1);

ClassProb = InitialClassProb;
PairProb = InitialPairProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.mu_y = [];
P.clg.sigma_y = [];
P.clg.mu_x = [];
P.clg.sigma_x = [];
P.clg.mu_angle = [];
P.clg.sigma_angle = [];
P.clg.theta = [];
% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  % Hint: This part should be similar to your work from PA8 and EM_cluster.m
  
  P.c = zeros(1,K);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  cProb = zeros(length(actionData),K);
  for i = 1:length(actionData)
      cProb(i,:) = cProb(i,:) + ClassProb(actionData(i).marg_ind(1),:);
  end
  cProb = mean(cProb,1);

  for k = 1:K
    P.c(k) = mean(cProb(:,k));%mean(ClassProb(:,k));%
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
  
  % M-STEP to estimate parameters for transition matrix
  % Fill in P.transMatrix, the transition matrix for states
  % P.transMatrix(i,j) is the probability of transitioning from state i to state j
  P.transMatrix = zeros(K,K);
  
  % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  P.transMatrix = P.transMatrix + size(PairProb,1) * .05;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  TM = sum(PairProb,1);
  
  P.transMatrix = P.transMatrix + reshape(TM,[K,K]);
  P.transMatrix = P.transMatrix./sum(P.transMatrix,2);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each 
  % of the poses in all actions = log( P(Pose | State) )
  % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m
  
  logEmissionProb = zeros(N,K);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  currDir = cd;
  altDir = '../wk3/';
  for i = 1:size(poseData,1)
      cd(altDir);
      for class = 1:K
        logEmissionProb(i,class) = ComputeProbability_Obs_given_class(P,G,poseData(i,:,:),class);
      end
      cd(currDir);
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP to compute expected sufficient statistics
  % ClassProb contains the conditional class probabilities for each pose in all actions
  % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
  % Also compute log likelihood of dataset for this iteration
  % You should do inference and compute everything in log space, only converting to probability space at the end
  % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues
  ClassProb = zeros(N,K);
  PairProb = zeros(V,K^2);
  loglikelihood(iter) = 0;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  pc = zeros(1,length(actionData));
  for i = 1:length(actionData)
      numStates = length(actionData(i).marg_ind);
      CTree = [struct('var',[1],'card',[K],...
                        'val',log(P.c) + logEmissionProb(actionData(i).marg_ind(1),:))];
      pc2 = CTree(1).val;
      for j = 2:numStates
          CTree(j) = struct('var',[j,j-1],'card',[K,K],...
                        'val',zeros(1,K*K));
          val = CTree(j).val;      
          for k1 = 1:K
              st = (k1-1)*K + 1;
              ed = k1*K;
              val(st:ed) = log(P.transMatrix(k1,:)) + logEmissionProb(actionData(i).marg_ind(j),:);
              pc2(k1) = pc2(k1) + logsumexp(val(st:ed));
          end
          CTree(j).val = val;
      end
      [M, PCalibrated] = ComputeExactMarginalsHMM(CTree);
      for j = 1:length(M)
          ClassProb(actionData(i).marg_ind(j),:) = exp(M(j).val);
          if j > 1
              pairProb1 = PCalibrated.cliqueList(j-1).val - logsumexp(PCalibrated.cliqueList(j-1).val);
              PairProb(actionData(i).pair_ind(j-1),:) = exp(pairProb1);
          end
          cd(currDir);
      end
%     ll = ComputeLogLikelihood(P,poseData(actionData(i).marg_ind,:,:),...
%         logEmissionProb(actionData(i).marg_ind,:));
%     S1 = FactorMarginalization(PCalibrated.cliqueList(1),[2]);
    loglikelihood(iter)  = loglikelihood(iter) + logsumexp(PCalibrated.cliqueList(1).val) + length(actionData(i).pair_ind);
    CTree = [];
  end
%   keyboard
%   loglikelihood(iter) = sum(pc);% + size(PairProb,1);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   LL = ComputeLogLikelihood(actionData,logEmissionProb,ClassProb,PairProb);
  % Print out loglikelihood
  if maxIter > 1
      disp(sprintf('EM iteration %d: log likelihood: %f', ...
        iter, loglikelihood(iter)));
      if exist('OCTAVE_VERSION')
        fflush(stdout);
      end
  end
  % Check for overfitting by decreasing loglikelihood
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
