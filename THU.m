function [weightVector,learningCurve]= ...
    THU(W,initialW,trainInput,trainTarget,a,stepSizeWeightVector,stepSizeTLS,flagLearningCurve,b)

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
    
    if n>100
     if abs(aprioriErr(n)/(sqrt((norm(weightVector)^2)+a)))>b
        alpha(n)=sqrt((norm(weightVector)^2)+a);
        weightVector = weightVector + b*sign(aprioriErr(n)/alpha(n))*(stepSizeWeightVector*trainInput(:,n)/alpha(n)+stepSizeWeightVector*aprioriErr(n)*weightVector/alpha(n)^3);
     else
        alpha(n)=aprioriErr(n) /(norm(weightVector)^2+a);
        weightVector = weightVector + stepSizeWeightVector*alpha(n)*(trainInput(:,n)+alpha(n)*weightVector);
     end
    
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
