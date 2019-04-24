% You should put all your code for recognizing unknown actions in this file.
% Describe the method you used in YourMethod.txt.
% Don't forget to call SavePrediction() at the end with your predicted labels to save them for submission, then submit using submit.m
function [predicted_labels] = RecognizeUnknownActions(datasetTrain, datasetTest, G, maxIter)

% INPUTS
% datasetTrain: dataset for training models, see PA for details
% datasetTest: dataset for testing models, see PA for details
% G: graph parameterization as explained in PA decription
% maxIter: max number of iterations to run for EM

% OUTPUTS
% accuracy: recognition accuracy, defined as (#correctly classified examples / #total examples)
% predicted_labels: N x 1 vector with the predicted labels for each of the instances in datasetTest, with N being the number of unknown test instances


% Train a model for each action
% Note that all actions share the same graph parameterization and number of max iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% numClasses = 3;
% for i = 1:length(datasetTrain)
%     CP_size = size(datasetTrain(i).InitialClassProb,1);
%     PP_size = size(datasetTrain(i).InitialPairProb,1);
%     CP = 0.05 + 0.9*rand(CP_size,numClasses);
%     PP = 0.05 + 0.9*rand(PP_size,numClasses^2);
%     CP = CP./sum(CP,2);
%     PP = PP./sum(PP,2);
%     datasetTrain(i).InitialClassProb = CP;
%     datasetTrain(i).InitialPairProb = PP;
% %     keyboard
% end
defStruct = struct();
defStruct.P = struct();
defStruct.ClassProb = 0;
defStruct.PairProb = 0;
defStruct.loglikelihood = 0;
classData = [defStruct;defStruct;defStruct];
for i = 1:length(datasetTrain)
    actionData = datasetTrain(i).actionData;
    poseData = datasetTrain(i).poseData;
    InitialClassProb = datasetTrain(i).InitialClassProb;
    InitialPairProb = datasetTrain(i).InitialPairProb;
    [P, loglikelihood,ClassProb,PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter);
    classData(i).P = P;
    classData(i).ClassProb = ClassProb;
    classData(i).PairProb = PairProb;
    classData(i).loglikelihood = loglikelihood;
    clearvars P ClassProb PairProb loglikehood;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Classify each of the instances in datasetTrain
% Compute and return the predicted labels and accuracy
% Accuracy is defined as (#correctly classified examples / #total examples)
% Note that all actions share the same graph parameterization

accuracy = 0;
predicted_labels = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j = 1:length(datasetTest.actionData)
    actionData = datasetTest.actionData(j);
    poseData = datasetTest.poseData(actionData.marg_ind,:,:);
    actionData.marg_ind = 1:length(actionData.marg_ind);
    actionData.pair_ind = 1:length(actionData.marg_ind)-1;
    for i = 1:length(classData)
        LL(i) = ComputeLogLikelihood(actionData,poseData,classData(i),G,size(classData(i).ClassProb,2));
    end
    predicted_labels(j) = find(LL == max(LL));
end

SavePredictions(predicted_labels');