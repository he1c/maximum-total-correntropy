function [weightVector,learningCurve]= ...
    MTC(W,initialW,trainInput,trainTarget,a,stepSizeWeightVector,stepSizeTLS,flagLearningCurve,kernelwidth)

% memeory initialization
[~,trainSize] = size(trainInput);

if flagLearningCurve
    learningCurve = zeros(trainSize,1);
else
    learningCurve = [];
end

weightVector = initialW;
biasTerm = 0;
aprioriErr = zeros(trainSize,1);
alpha = zeros(trainSize,1);
MCCkernel=1/2/kernelwidth^2;
% training
for n = 1:trainSize
    networkOutput = weightVector'*trainInput(:,n) + biasTerm;
    aprioriErr(n) = trainTarget(n) - networkOutput;    
    
    if n>100
    alpha(n)=sqrt((norm(weightVector)^2)+a);
    weightVector = weightVector + exp(-((aprioriErr(n)/alpha(n))^2)*MCCkernel)*(aprioriErr(n)/alpha(n))*(stepSizeWeightVector*trainInput(:,n)/alpha(n)+stepSizeWeightVector*aprioriErr(n)*weightVector/alpha(n)^3);
    
    else
    alpha(n)=aprioriErr(n) /(norm(weightVector)^2+a);
    weightVector = weightVector + stepSizeTLS*alpha(n)*(trainInput(:,n)+alpha(n)*weightVector);
    end
    if flagLearningCurve
        err = weightVector-W;
        learningCurve(n) = sum(err.^2);
    end
end

return
