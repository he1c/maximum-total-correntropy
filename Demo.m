clear

MC=100;

varNoiseA = 0.04;
varNoiseB = 9;
c=0.0;
a=1;

inputDimension = 4; 
inputsize = 5000;

W=[0.4,0.7,-0.3,0.5]';
kernelwidth=0.5;
 
u=randn(inputsize,1);

x = zeros(inputDimension,inputsize-3);
for k = 1:inputsize-3
    x(:,k) = u(k:k+inputDimension-1);
end

y=x'*W;

L=length(y);

for jj=1:1:3

    ensembleLearningCurveglms=zeros(L,1);
    ensembleLearningCurvegmcc=zeros(L,1);
    ensembleLearningCurvegtls=zeros(L,1);
    ensembleLearningCurvegtmcc=zeros(L,1);
    ensembleLearningCurvegthu=zeros(L,1);
    ensembleLearningCurvegtl1=zeros(L,1);
    ensembleLearningCurvegtf=zeros(L,1);

    if(jj==1)
        c=0;stepsize = [0.009,0.018,0.01,0.02,0.0032,0.018,0.012];
    elseif(jj==2)
        c=0.03;varNoiseB = 2;stepsize = [0.009,0.018,0.009,0.02,0.0032,0.018,0.012];
    elseif(jj==3)  
        c=0.08;varNoiseB = 9;stepsize = [0.007,0.02,0.006,0.027,0.0032,0.018,0.008]; 
    elseif(jj==4)  
        c=0.03;varNoiseB = 9;stepsize = [0.008,0.015,0.007,0.022,0.0032,0.018,0.012]; 
    end

    for mc = 1:MC

        initialW=0.1*randn(inputDimension,1);

        vin=noisemix(L,4,c,varNoiseA,varNoiseB,'gaussian')';

        trainInput=x+vin;

        v=noisemix(L,1,c,varNoiseA*a,varNoiseB,'gaussian')';

        trainTarget=y+v';

        noise_ratio=var(v)/var(vin(1,:));

        [weightlms,learningCurvelms] = LMS(W,initialW,trainInput,trainTarget,stepsize(1),1); %0.006
        [weightmcc,learningCurvemcc] = MCC(W,initialW,trainInput,trainTarget,stepsize(2),stepsize(1),1,kernelwidth); %0.014
        [weighttls,learningCurvetls] = TLS(W,initialW,trainInput,trainTarget,a,stepsize(3),1); %0.003*2
        [weightmtc,learningCurvetmcc] = MTC(W,initialW,trainInput,trainTarget,a,stepsize(4),stepsize(3),1,kernelwidth*sqrt(2)/2); %0.007
        [weighttl1,learningCurvetl1] = TL1(W,initialW,trainInput,trainTarget,a,stepsize(5),stepsize(3),1); %0.007
        [weightthu,learningCurvethu] = THU(W,initialW,trainInput,trainTarget,a,stepsize(6),stepsize(3),1,0.2); %0.007
        [weighttf,learningCurvetf] = TF(W,initialW,trainInput,trainTarget,a,stepsize(7),stepsize(3),1,1.3998); %0.007

        ensembleLearningCurveglms=ensembleLearningCurveglms+learningCurvelms;
        ensembleLearningCurvegmcc=ensembleLearningCurvegmcc+learningCurvemcc;
        ensembleLearningCurvegtls=ensembleLearningCurvegtls+learningCurvetls;
        ensembleLearningCurvegtmcc=ensembleLearningCurvegtmcc+learningCurvetmcc;
        ensembleLearningCurvegtl1=ensembleLearningCurvegtl1+learningCurvetl1;
        ensembleLearningCurvegthu=ensembleLearningCurvegthu+learningCurvethu;
        ensembleLearningCurvegtf=ensembleLearningCurvegtf+learningCurvetf;

    end


    figure,plot(10*log10(ensembleLearningCurveglms/MC),'linewidth',1.5)
    hold on;
    plot(10*log10(ensembleLearningCurvegmcc/MC),'linewidth',1.5)
    plot(10*log10(ensembleLearningCurvegtls/MC),'linewidth',1.5)
    plot(10*log10(ensembleLearningCurvegthu/MC),'linewidth',1.5)
    plot(10*log10(ensembleLearningCurvegtl1/MC),'linewidth',1.5)
    plot(10*log10(ensembleLearningCurvegtf/MC),'linewidth',1.5)
    plot(10*log10(ensembleLearningCurvegtmcc/MC),'linewidth',1.5)
    hold off
    xlabel('iteration'),ylabel('MSD(dB)')

    grid on
    legend(['LMS \mu=' num2str(stepsize(1))],['MCC \mu=' num2str(stepsize(2))],['GD-TLS \mu=' num2str(stepsize(3))],['THU \mu=' num2str(stepsize(6))],['TLAD \mu=' num2str(stepsize(5))],['TF \mu=' num2str(stepsize(7))],['MTC \mu=' num2str(stepsize(4))]);

end

