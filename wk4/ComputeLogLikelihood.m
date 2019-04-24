function loglikelihood = ComputeLogLikelihood(actionData,poseData,classData,G,K)

P = classData.P;
currDir = cd;
altDir = '../wk3/';
cd(altDir);
for i = 1:size(poseData,1)
    for class = 1:K
        logEmissionProb(i,class) = ComputeProbability_Obs_given_class(P,G,poseData(i,:,:),class);
    end
end
cd(currDir);

CTree = [struct('var',[1],'card',[K],'val',log(P.c) + logEmissionProb(actionData.marg_ind(1),:))];
numStates = length(actionData.marg_ind);
for j = 2:numStates
    CTree(j) = struct('var',[j,j-1],'card',[K,K],'val',zeros(1,K*K));
    val = CTree(j).val;
    for k1 = 1:K
        st = (k1-1)*K + 1;
        ed = k1*K;
        val(st:ed) = log(P.transMatrix(k1,:)) + logEmissionProb(actionData.marg_ind(j),:);
    end
    CTree(j).val = val;
end
[M, PCalibrated] = ComputeExactMarginalsHMM(CTree);
loglikelihood  = logsumexp(PCalibrated.cliqueList(1).val) + length(actionData.pair_ind);