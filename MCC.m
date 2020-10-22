function [weightVector,learningCurve]= ...
    MCC(W,initialW,trainInput,trainTarget,stepSizeWeightVector,stepSizeLMS,flagLearningCurve,kernelwidth)

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
MCCkernel=1/2/kernelwidth^2;

% training
for n = 1:trainSize
    networkOutput = weightVector'*trainInput(:,n) + biasTerm;
    aprioriErr(n) = trainTarget(n) - networkOutput;
    if n>100
        weightVector = weightVector + stepSizeWeightVector*exp(-(aprioriErr(n)^2)*MCCkernel)*aprioriErr(n)*trainInput(:,n);
    else
        weightVector = weightVector + stepSizeLMS*aprioriErr(n)*trainInput(:,n);
    end
    if flagLearningCurve
        err = weightVector-W;
        learningCurve(n) = sum(err.^2);
    end
end

return
