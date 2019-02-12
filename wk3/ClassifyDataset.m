function accuracy = ClassifyDataset(dataset, labels, P, G)
% returns the accuracy of the model P and graph G on the dataset 
%
% Inputs:
% dataset: N x 10 x 3, N test instances represented by 10 parts
% labels:  N x 2 true class labels for the instances.
%          labels(i,j)=1 if the ith instance belongs to class j 
% P: struct array model parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description) 
%
% Outputs:
% accuracy: fraction of correctly classified instances (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
accuracy = 0.0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class = length(P.c);
prediction = zeros(size(labels));
for k = 1:size(dataset,1)
    log_prob_ObsGClass = zeros(1,class);
    prob_Obs_and_Class = zeros(1,class);
    for i = 1:class
        log_prob_ObsGClass(i) = ComputeProbability_Obs_given_class(P,G,dataset(k,:,:),i);
        prob_Obs_and_Class(i) = log(P.c(i)) + log_prob_ObsGClass(i);
    end
    ind = find(prob_Obs_and_Class == max(prob_Obs_and_Class));
    prediction(k,ind) = 1;
end

diff_ind = find(prediction(:,1)-labels(:,1) == 0);
accuracy = length(diff_ind)/size(labels,1);

fprintf('Accuracy: %.2f\n', accuracy);