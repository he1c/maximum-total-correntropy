function [weightVector,learningCurve]= ...
    TLS(W,initialW,trainInput,trainTarget,a,stepSizeWeightVector,flagLearningCurve)

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
alpha = zeros(trainSize,1);

% training
for n = 1:trainSize
    networkOutput = weightVector'*trainInput(:,n) + biasTerm;
    aprioriErr(n) = trainTarget(n) - networkOutput;
    alpha(n)=aprioriErr(n) /(norm(weightVector)^2+a);
    weightVector = weightVector + stepSizeWeightVector*alpha(n)*(trainInput(:,n)+alpha(n)*weightVector);
    if flagLearningCurve
        err = weightVector-W;
        learningCurve(n) = sum(err.^2);
    end
end

return
