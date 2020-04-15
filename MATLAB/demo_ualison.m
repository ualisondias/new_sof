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
load dados_certo.mat
tic
%for i=12:1:15; %Encontrar melhor valor de Granularidade
%for j=12:1:15; %Encontrar melhor valor de Granularidade
i=1;
for ii=1:33
    for iii=1:5
%% O Classificador SOF conduzindo aprendizado offline apartir de dados estáticos
Input.TrainingData=Dados.dados(ii).dados(iii).DTra2;
Input.TrainingLabel=Dados.dados(ii).dados(iii).LTra2;
GranLevel=i;
%DistanceType='Cosine';
%DistanceType='Euclidean';
%DistanceType='Minkowski';
%DistanceType='Mahalanobis'; 
DistanceType='Hamming';
Mode='OfflineTraining';
[Output0]=SOFClassifier_ualison(Input,GranLevel,Mode,DistanceType);
%% O Classificador SOF conduzindo aprendizado online apartir de transmissão de dados depois de serem preparados offline
Input=Output0;               
Input.TrainingData=Dados.dados(ii).dados(iii).DTra1;    
Input.TrainingLabel=Dados.dados(ii).dados(iii).LTra1;   
Mode='EvolvingTraining';
[Output1]=SOFClassifier_ualison(Input,GranLevel,Mode,DistanceType);
Output1.TrainedClassifier;
Output1.EstimatedLabel;
matriz_confusao_evo=Output1.ConfusionMatrix;     
max_colu_evo=max([matriz_confusao_evo]);
con_colu_evo=sum(matriz_confusao_evo');
Acc_evo=max_colu_evo./con_colu_evo;
mean_Acc_evo=mean(Acc_evo);
[Result_evo,RefereceResult_evo]=confusion.getValues(matriz_confusao_evo);
acura_total_treino(ii,iii)=Result_evo.Accuracy;
F1_score_total_treino(ii,iii)=Result_evo.F1_score;
Kappa_total_treino(ii,iii)=Result_evo.Kappa;
mse_total_treino(ii,iii)=immse(Dados.dados(ii).dados(iii).LTra1,Output1.EstimatedLabel);





%% O classidicador SOF conduzindo as validações com os dados de teste
Input=Output1;
Input.TestingData=Dados.dados(ii).dados(iii).DTes1;
Input.TestingLabel=Dados.dados(ii).dados(iii).LTes1;
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
%mean_Acc(j)=mean(Acc);
%i
[Result,RefereceResult]=confusion.getValues(matriz_confusao);
%disp(Result);
Acuracia_teste=Result.Accuracy;
%Acuracia(j)=Result.Accuracy
%end %Encontrar melhor valor de Granularidade
%% End of example 
acura_total_teste(ii,iii)=Acuracia_teste
F1_score_total_teste(ii,iii)=Result.F1_score;
Kappa_total_teste(ii,iii)=Result.Kappa;
passada=length(acura_total_teste)
mse_total_teste(ii,iii)=immse(Dados.dados(ii).dados(iii).LTes1,Output2.EstimatedLabel);
%media(iii,1)=mean(acura_total(:,iii));
%desvio(iii,1)=std(acura_total(:,iii))*100;
    end
end
toc
espaco=zeros(33,2);
resultado_treino=[acura_total_treino espaco F1_score_total_treino espaco Kappa_total_treino espaco mse_total_treino espaco];
resultado_teste=[acura_total_teste espaco F1_score_total_teste espaco Kappa_total_teste espaco mse_total_teste espaco];
resultado_todos=[resultado_treino resultado_teste];
save resultado_todos resultado_todos;

