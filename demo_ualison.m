 %% This code is the Self-Organising Fuzzy Logic (SOF) classifier
 clear all
 clc
 close all
% %% Example 1
% load letter1.mat
% %% O Classificador SOF conduzindo aprendizado offline apartir de dados estáticos
% Input.TrainingData=DTra1;    % Input data samples
% Input.TrainingLabel=LTra1;   % Labels of the input data samples
% GranLevel=12;                % Level of granularity (Once being fixed in offline training stage, it cannot be changed further)
% DistanceType='Hamming';  % Type of distance/dissimilarity SOF classifier uses, which can be 'Mahalanobis', 'Cosine' or 'Euclidean'
% Mode='OfflineTraining';      % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
% [Output1]=SOFClassifier_ualison(Input,GranLevel,Mode,DistanceType); 
% %Output1.TrainedClassifier  %- Offline primed SOF classifier
% % Output2.EstimatedLabel;      %- Estimated label of validation data
% %Output1.ConfusionMatrix     %- Confusion matrix of the result
% %% O classificador SOF conduzindo validação dos dados de teste
% Input=Output1;               % Offline primed SOF classifier
% Input.TestingData=DTes1;     % Testing 
% Input.TestingLabel=LTes1;    % Labels of the tetsing data samples
% Mode='Validation';           % Operating mode, which can be 'OfflineTraining', 'EvolvingTraining' or 'Validation'
% [Output2]=SOFClassifier_ualison(Input,GranLevel,Mode,DistanceType);
% Output2.TrainedClassifier;  %- Trained SOF classifier (same as the input)
% Output2.EstimatedLabel;      %- Estimated label of validation data
% Output2.ConfusionMatrix     %- Confusion matrix of the result
% matriz_confusao=Output2.ConfusionMatrix;     %- Confusion matrix of the result
% max_colu=max([matriz_confusao]);
% con_colu=sum(matriz_confusao');
% Acc=max_colu./con_colu;
% mean_Acc=mean(Acc)
% % End of example 1


%% Example 2
load frequence2.mat
%for i=1:1:20; %Encontrar melhor valor de Granularidade
i=14;
%% O Classificador SOF conduzindo aprendizado offline apartir de dados estáticos
Input.TrainingData=DTra2;
Input.TrainingLabel=LTra2;
GranLevel=i;
%DistanceType='Cosine';
%DistanceType='Euclidean';
DistanceType='Minkowski';
%DistanceType='Mahalanobis'; 
%DistanceType='Hamming';
Mode='OfflineTraining';
tic
[Output0]=SOFClassifier_ualison(Input,GranLevel,Mode,DistanceType);
%% O Classificador SOF conduzindo aprendizado online apartir de transmissão de dados depois de serem preparados offline
Input=Output0;               
Input.TrainingData=DTra1;    
Input.TrainingLabel=LTra1;   
Mode='EvolvingTraining';
[Output1]=SOFClassifier_ualison(Input,GranLevel,Mode,DistanceType);
Output1.TrainedClassifier;
% Output1.EstimatedLabel;
% matriz_confusao_evo=Output1.ConfusionMatrix;     
% max_colu_evo=max([matriz_confusao_evo]);
% con_colu_evo=sum(matriz_confusao_evo');
% Acc_evo=max_colu_evo./con_colu_evo;
% mean_Acc=mean(Acc);
%% O classidicador SOF conduzindo as validações com os dados de teste
Input=Output1;
Input.TestingData=DTes1;
Input.TestingLabel=LTes1;
Mode='Validation';
[Output2]=SOFClassifier_ualison(Input,GranLevel,Mode,DistanceType);
Output2.TrainedClassifier;  
Output2.EstimatedLabel;      
matriz_confusao=Output2.ConfusionMatrix;     
max_colu=max([matriz_confusao]);
con_colu=sum(matriz_confusao');
Acc=max_colu./con_colu;
mean_Acc=mean(Acc);
%mean_Acc(i)=mean(Acc);
%i
[Result,RefereceResult]=confusion.getValues(matriz_confusao);
MSE = sum((Output2.EstimatedLabel-LTes1).^2)/length(Output2.EstimatedLabel);
err = immse(LTes1,Output2.EstimatedLabel);
disp(Result)
toc

%end %Encontrar melhor valor de Granularidade
%% End of example 2