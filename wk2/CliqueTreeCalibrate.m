%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   [P] = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-product
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%  
%   [P, logZ] = CLIQUETREECALIBRATE(P, isMax) also returns the log partition function
%   corresponding to P. Note that 'isMax' must be false if you want to
%   compute logZ.
%   YOU NEED TO IMPLEMENT THIS. 
%   Hint: You may find the unnormalizedMessages array useful (it's new to this assignment).

% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P, logZ] = CliqueTreeCalibrate(P, isMax)

    % number of cliques in the tree.
    N = length(P.cliqueList);

    if (nargout > 1)
        if (isMax)
            error ('Cannot compute logZ when performing max-product calibration');
        end
        unnormalizedMessages = repmat(struct('var', [], 'card', [], 'val', []), N, N);
        doLogZ = true;
    else
        doLogZ = false;
    end


    % setting up the messages that will be passed.
    messages = repmat(struct('var', [], 'card', [], 'val', []), N, N);

    if isMax == 1
        % Convert factors to log space
        for i = 1 : length(P.cliqueList)
            % Iterate through cliques
            P.cliqueList(i).val = log(P.cliqueList(i).val);
        end
    end

    % These will make sure that the same message is not passed twice in a row
    lastCliqueOne = 0;
    lastCliqueTwo = 0;

    while (1), 
        % Find the clique that is ready, compute the message for that clique,
        % and add that message to the queue
        [cliqueOne, cliqueTwo] = GetNextCliques(P, messages);
        if (cliqueOne == 0) || (cliqueTwo == 0)
            % There are no ready cliques, so stop
            break
        end
        if (lastCliqueOne == cliqueOne) && (lastCliqueTwo == cliqueTwo)
            % Do not pass the same message twice in a row, so stop
            break
        end
        lastCliqueOne = cliqueOne;
        lastCliqueTwo = cliqueTwo;
        currentMessage = P.cliqueList(cliqueOne);
        if (doLogZ)
            currentUnnormalizedMessage = P.cliqueList(cliqueOne);
        end
        for i = 1:N
            % Iterate through edges and compute the outgoing message
            if i == cliqueTwo
                % Do not consider this edge
                continue
            end
            if isMax == 0
                % Compute the factor product
                currentMessage = FactorProduct(currentMessage, messages(i, cliqueOne));
                if (doLogZ)
                    currentUnnormalizedMessage = FactorProduct(currentUnnormalizedMessage, unnormalizedMessages(i, cliqueOne));
                end            
            else
                % Compute the sum because doing max-sum message passing
                % (summing the logs)
                currentMessage = FactorSum(currentMessage, messages(i, cliqueOne));
            end
        end
        varsToRemove = setdiff(P.cliqueList(cliqueOne).var, P.cliqueList(cliqueTwo).var);
        if isMax == 0
            % Doing sum-product
            messages(cliqueOne,cliqueTwo) = FactorMarginalization(currentMessage, varsToRemove);
            messages(cliqueOne,cliqueTwo).val = messages(cliqueOne,cliqueTwo).val/sum(messages(cliqueOne,cliqueTwo).val);
            if (doLogZ)
                unnormalizedMessages(cliqueOne, cliqueTwo) = FactorMarginalization(currentUnnormalizedMessage, varsToRemove);
            end
        else
            % Use FactorMaxMarginalization to get the marginals
            messages(cliqueOne,cliqueTwo) = FactorMaxMarginalization(currentMessage, varsToRemove);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end

%%    keyboard
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % If doLogZ is set, you need to use the unnormalizedMessages to compute
    % logZ, the log of the partition function.
    if (doLogZ)
        %%% YOUR CODE HERE:
        mu = 1;
        for i = 1:size(P.edges,1)
          ind = find(P.edges(i,i:end) == 1);
          for j = 1:length(ind)
            mu = mu.*unnormalizedMessages(i,ind(j)).val.*unnormalizedMessages(ind(j),i).val;
          end
        end
        logZ = log(sum(mu));
          
            
        %%% Brute Force method
%        logZ = 0; % remove this
%        allVars = [];
%        allCard = [];
%        for i = 1:N
%          allVars = [allVars,P.cliqueList(i).var];
%          allCard = [allCard,P.cliqueList(i).card];
%        end
%        [uniqueVars,I] = unique(allVars);
%        uniqueCard = allCard(I);
%        assgn = IndexToAssignment(1:prod(uniqueCard),uniqueCard);
%        idx = cell(N,1);
%        for j = 1:N
%          idx_temp = [];
%          for k = 1:length(P.cliqueList(j).var)
%            idx_temp(k) = find(uniqueVars == P.cliqueList(j).var(k));
%          end
%          idx{j} = idx_temp;
%        end
%        for i = 1:size(assgn,1)
%          Z = 1.0;
%          A = assgn(i,:);
%          for j = 1:N
%            cliqueA = A(idx{j});
%            idx1 = AssignmentToIndex(cliqueA,P.cliqueList(j).card);
%            Z = Z*P.cliqueList(j).val(idx1);
%          end
%%          [Z,log(Z)]
%          logZ = logZ + Z;
%        end
%        logZ = log(logZ);
    else
        logZ = 0;
    end
%   keyboard
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Iterate through the incoming messages, multiply them by the initial
    % potential, and normalize
    for i = 1:N
        % Iterate through cliques and find the final potentials
        for j = 1:N
            % Iterate through messages and multiply them
            if isMax == 0
                % Compute the factor product
                P.cliqueList(i) = FactorProduct(P.cliqueList(i), messages(j,i));
            else
                % Compute the factor sum because doing max-sum message passing
                % (need to sum the logs)
                P.cliqueList(i) = FactorSum(P.cliqueList(i), messages(j,i));
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
