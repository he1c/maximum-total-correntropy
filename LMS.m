function [weightVector,learningCurve]= ...
    LMS(W,initialW,trainInput,trainTarget,stepSizeWeightVector,flagLearningCurve)

% memeory initialization
[inputDimension,trainSize] = size(trainInput);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
else
    learningCurve = [];
end

weightVector = initialW;
biasTerm = 0;
aprioriErr = zeros(trainSize,1);
WeightR=zeros(trainSize,inputDimension);

% training
for n = 1:trainSize
    networkOutput = weightVector'*trainInput(:,n) + biasTerm;
    aprioriErr(n) = trainTarget(n) - networkOutput;
    weightVector = weightVector + stepSizeWeightVector*aprioriErr(n)*trainInput(:,n);
    WeightR(n,:)=weightVector;
    if flagLearningCurve
        err = weightVector-W;
        learningCurve(n) = sum(err.^2);
    end
end

return
