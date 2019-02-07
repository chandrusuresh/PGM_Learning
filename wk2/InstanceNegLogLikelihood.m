% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y). 
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (numParams x 1 vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)

    % featureSet is a struct with two fields:
    %    .numParams - the number of parameters in the CRF (this is not numImageFeatures
    %                 nor numFeatures, because of parameter sharing)
    %    .features  - an array comprising the features in the CRF.
    %
    % Each feature is a binary indicator variable, represented by a struct 
    % with three fields:
    %    .var          - a vector containing the variables in the scope of this feature
    %    .assignment   - the assignment that this indicator variable corresponds to
    %    .paramIdx     - the index in theta that this feature corresponds to
    %
    % For example, if we have:
    %   
    %   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
    %
    % then feature is an indicator function over X_2 and X_3, which takes on a value of 1
    % if X_2 = 5 and X_3 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
    % Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
    %
    % If you're interested in the implementation details of CRFs, 
    % feel free to read through GenerateAllFeatures.m and the functions it calls!
    % For the purposes of this assignment, though, you don't
    % have to understand how this code works. (It's complicated.)
    
    featureSet = GenerateAllFeatures(X, modelParams);

    % Use the featureSet to calculate nll and grad.
    % This is the main part of the assignment, and it is very tricky - be careful!
    % You might want to code up your own numerical gradient checker to make sure
    % your answers are correct.
    %
    % Hint: you can use CliqueTreeCalibrate to calculate logZ effectively. 
    %       We have halfway-modified CliqueTreeCalibrate; complete our implementation 
    %       if you want to use it to compute logZ.
    
    nll = 0;
    grad = zeros(size(theta));
    %%%
    % Your code here:
    % % % Weighted Feature Counts
    wfc = 0;
    grad_wfc = zeros(1,length(theta));
    for i = 1:length(featureSet.features)
        if y(featureSet.features(i).var) == featureSet.features(i).assignment
            wfc = wfc + theta(featureSet.features(i).paramIdx);
            grad_wfc(featureSet.features(i).paramIdx) = 1;
        end
    end
    
    % % % Regularization cost
    rc = modelParams.lambda/2*(sum(theta.^2));
    grad_rc = modelParams.lambda*theta;
    
    % % % Clique Tree
    cliqueTree = struct();
    cliqueTree.edges = [0,1;1,0];
    
    % % % Get independent factors
    indpFactors = [struct('var',[1],'card',[26],'val',zeros(1,26),'idx',[]),...
                   struct('var',[2],'card',[26],'val',zeros(1,26),'idx',[]),...
                   struct('var',[3],'card',[26],'val',zeros(1,26),'idx',[])];
    indpFactors(1).idx = cell(1,26);
    indpFactors(2).idx = cell(1,26);
    indpFactors(3).idx = cell(1,26);               
               
    for i = 1:length(featureSet.features)
        if length(featureSet.features(i).var) == 1
            varNum = featureSet.features(i).var;
            valNum = featureSet.features(i).assignment;
            paramIdx = featureSet.features(i).paramIdx;
            indpFactors(varNum).val(valNum) = indpFactors(varNum).val(valNum)+theta(paramIdx);
            indpFactors(varNum).idx{valNum} = [indpFactors(varNum).idx{valNum},paramIdx];
        end
    end
    
    depFactors = [struct('var',[1,2],'card',[26,26],'val',zeros(1,26*26)),...
                  struct('var',[2,3],'card',[26,26],'val',zeros(1,26*26))];
    depFactors_idx = cell(1,2);
    for i = 1:length(featureSet.features)
        if length(featureSet.features(i).var) ~= 1
            if prod(featureSet.features(i).var == [1,2])
                varNum = 1;
            elseif prod(featureSet.features(i).var == [2,3])
                varNum = 2;
            else
                continue;
            end
            valNum = AssignmentToIndex(featureSet.features(i).assignment,depFactors(varNum).card);
            val = theta(featureSet.features(i).paramIdx) + ...
                indpFactors(depFactors(varNum).var(2)).val(featureSet.features(i).assignment(2));
            if varNum == 1
                val = val + ...
                    indpFactors(depFactors(varNum).var(1)).val(featureSet.features(i).assignment(1));
            end
            depFactors(varNum).val(valNum) = exp(val);
            depFactors_idx{varNum}(valNum) = featureSet.features(i).paramIdx;
        end
    end
    cliqueTree.cliqueList = depFactors;
    [calibratedCTree, logZ] = CliqueTreeCalibrate(cliqueTree, false);
    % % % Normalize calibrated clique tree
    calibratedCTree_norm = calibratedCTree;
    calibratedCTree_norm.cliqueList(1).val = calibratedCTree_norm.cliqueList(1).val/sum(calibratedCTree_norm.cliqueList(1).val);
    calibratedCTree_norm.cliqueList(2).val = calibratedCTree_norm.cliqueList(2).val/sum(calibratedCTree_norm.cliqueList(2).val);
    
    % % % Individual Probabilities
    P{1} = FactorMarginalization(calibratedCTree_norm.cliqueList(1),[2]);
    P{2} = FactorMarginalization(calibratedCTree_norm.cliqueList(1),[1]);
    P2_2 = FactorMarginalization(calibratedCTree_norm.cliqueList(2),[3]);
    P{3} = FactorMarginalization(calibratedCTree_norm.cliqueList(2),[2]);
    
    % % Check if P2_1 = P2_2 because calibration is supposed to ensure
    % % this!
    P2_val = sum(P{2}.val-P2_2.val);
    assert(P2_val <= 1E-10);
    
    grad_factor = zeros(1,length(theta));
    
    for i = 1:length(indpFactors)
        for j = 1:length(indpFactors(i).idx)
            grad_factor(indpFactors(i).idx{j}) = grad_factor(indpFactors(i).idx{j}) + ...
                P{i}.val(j);
        end
    end
    for i = 1:length(depFactors_idx)
        for j = 1:length(depFactors_idx{i})
            grad_factor(depFactors_idx{i}(j)) = grad_factor(depFactors_idx{i}(j)) + ...
                calibratedCTree_norm.cliqueList(i).val(j);
        end
    end    
    
    nll = logZ - wfc + rc;

    % % % Gradient computation
    grad = grad_factor - grad_wfc + grad_rc;
end
