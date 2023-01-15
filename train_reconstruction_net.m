% Training of neural network for image reconstruction of digits propagated 
% through multimode fiber

clear all
close all


%% Load training data
%enthaelt XTrain, YTrain, XValid, YValid, XTest, YTest 

%fuer Aufgabe 1
load("DATA_MMF_16.mat");


%nur fuer Verstaendnis
whos XTrain;
whos YTrain;
whos XValid;
whos YValid;
whos XTest;
whos YTest;


%% Create Neural Network Layergraph MLP
%Aufloesung Eingangs- und Ausgangsbilder
ipx = size(XTrain, 1);
opx = size(YTrain, 1);

%nur fuer Verstaendnis
whos ipx
whos opx


layers = [imageInputLayer([ipx ipx 1],'Name','Input')
    fullyConnectedLayer(ipx^2,'Name','Fc1')
    reluLayer('Name','Relu1')
    fullyConnectedLayer(opx^2,'Name','Fc2')
    reluLayer('Name','Relu2')
    depthToSpace2dLayer([opx opx],'Name','dts1')
    regressionLayer('Name','Output')
    ];

analyzeNetwork(layers);

%% Training network
% define "trainingOptions"

options = trainingOptions("adam");

options.MiniBatchSize = 64;

options.MaxEpochs = 50;

options.InitialLearnRate = 0.001;

options.ExecutionEnvironment = 'auto';

options.OutputNetwork = 'best-validation-loss';

options.ValidationData = {XValid, YValid};

options.Plots = 'training-progress';

options.ValidationPatience = 20;

%% training using "trainNetwork"
% ohne augmentationdata
mlp = trainNetwork(XTrain,YTrain,layers,options);

%% fuer Aufgabe 2/3
load("DATA_MMF_16_aug.mat")

% mit augmentationdata 
Augmlp = trainNetwork(XTrain,YTrain,layers,options);

%% Calculate Prediction 
%Aufgabe 3
% use command "predict"

% ohne augmentationdata
YPred = predict(mlp,XTest);

% mit augmentationdata
AugYPred = predict(Augmlp,XTest);


%% Evaluate Network
%Aufgabe 3
% calculate RMSE, Correlation, SSIM, PSNR

% ohne augmentationdata
rmse_tmp = rmse(YPred(),single(YTest()),[1 2]);

for i = 1 : size(YPred,4) 
    mlp_ssim(i) = ssim(YPred(:,:,1, i),single(YTest(:,:,1,i)));
    mlp_psnr(i) = psnr(YPred(:,:,1, i),single(YTest(:,:,1,i)));    
    mlp_corr(i) = corr2(YPred(:,:,1, i),single(YTest(:,:,1,i)));
    mlp_rmse(i) = rmse_tmp(:,:,1,i);
end

% mit augmentationdata
aug_rmse_tmp = rmse(YPred(),single(YTest()),[1 2]);

for i = 1 : size(YPred,4) 
    Augmlp_ssim(i) = ssim(AugYPred(:,:,1, i),single(YTest(:,:,1,i)));
    Augmlp_psnr(i) = psnr(AugYPred(:,:,1, i),single(YTest(:,:,1,i)));    
    Augmlp_corr(i) = corr2(AugYPred(:,:,1, i),single(YTest(:,:,1,i)));
    Augmlp_rmse(i) = aug_rmse_tmp(:,:,1,i);
end


%% Boxplots for step 6 of instructions
%Aufgabe 3

figure
%RMSE
subplot(2,2,1)
boxchart([mlp_rmse; Augmlp_rmse]')
title('RMSE')
ylabel('RMSE')
legend("1:MLP 2:AugMLP");
%CORR
subplot(2,2,2)
boxchart([mlp_corr; Augmlp_corr]') 
title('Correlation')
ylabel('Correlation')
legend("1:MLP 2:AugMLP") 
%PSNR
subplot(2,2,3)
boxchart([mlp_psnr; Augmlp_psnr]')
title('PSNR')
ylabel('PSNR')
legend("1:MLP 2:AugMLP") 
%SSIM
subplot(2,2,4)
boxchart([mlp_ssim; Augmlp_ssim]')
title('SSIM')
ylabel('SSIM')
legend("1:MLP 2:AugMLP") 


%% Step 7: create Neural Network Layergraph U-Net
%Aufgabe 4

Layers = unetLayers([ipx ipx 1],2,'encoderDepth',3);
 
finalConvLayer = convolution2dLayer(1,1,'Padding','same','Stride',1,'Name','Final-ConvolutionLayer');
Layers = replaceLayer(Layers,'Final-ConvolutionLayer',finalConvLayer);

Layers = removeLayers(Layers,'Softmax-Layer');

regLayer = regressionLayer('Name','Reg-Layer');
Layers = replaceLayer(Layers,'Segmentation-Layer',regLayer);

Layers = connectLayers(Layers,'Final-ConvolutionLayer','Reg-Layer');



analyzeNetwork(Layers);



%% Train Unet
unet = trainNetwork(XTrain,YTrain,Layers,options);


%% Boxplots for step 8 of instructions
%Aufgabe 5
UYPred = predict(unet,XTest);

unet_rmse_tmp = rmse(UYPred(),single(YTest()),[1 2]);

for i = 1 : size(UYPred,4)
    unet_ssim(i) = ssim(UYPred(:,:,1, i),single(YTest(:,:,1,i)));
    unet_psnr(i) = psnr(UYPred(:,:,1, i),single(YTest(:,:,1,i)));    
     unet_corr(i) = corr2(UYPred(:,:,1, i),single(YTest(:,:,1,i)));
     unet_rmse(i) = unet_rmse_tmp(:,:,1,i);
end

figure
%RSME
subplot(2,2,1)
boxchart([unet_rmse; Augmlp_rmse]')
title('RMSE')
ylabel('RMSE')
legend("1:Unet 2:AugMLP");
%CORR
subplot(2,2,2)
boxchart([unet_corr; Augmlp_corr]') 
title('Correlation')
ylabel('Correlation')
legend("1:Unet 2:AugMLP") 
%PSNR
subplot(2,2,3)
boxchart([unet_psnr; Augmlp_psnr]')
title('PSNR')
ylabel('PSNR')
legend("1:Unet 2:AugMLP") 
%SSIM
subplot(2,2,4)
boxchart([unet_ssim; Augmlp_ssim]')
title('SSIM')
ylabel('SSIM')
legend("1:Unet 2:AugMLP") 



