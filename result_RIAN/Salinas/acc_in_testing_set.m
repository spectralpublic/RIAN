clc;
clear;
close all;

% 可以在这里更改输入，以计算不同旋转角度下的定量指标，
% prob_map_0_9_199.mat 为测试样本旋转0度，9表示训练和测试时高光谱图像块的空间尺寸为9x9
% prob_map_90_9_199.mat 为测试样本旋转90度，9表示训练和测试时高光谱图像块的空间尺寸为9x9
% prob_map_180_9_199.mat 为测试样本旋转180度，9表示训练和测试时高光谱图像块的空间尺寸为9x9
% prob_map_270_9_199.mat 为测试样本旋转270度，9表示训练和测试时高光谱图像块的空间尺寸为9x9
load('prob_map_180_9_199.mat')

load('Salinas_gt.mat')
load('TestIndex_9.mat')
load('TrainIndex_9.mat')

%% GT 可视化ground truth
Label = salinas_gt;
Label_GT_map =label2color(Label,'india');
figure
imshow(Label_GT_map)
title('Ground Truth')
axis image;set(gca,'xtick',[],'ytick',[]);

%% training set 可视化训练集
label = Label(:);
[H,W] = size(Label);
TrainIndex  = TrainIndex + 1;
TrainingMap = zeros(H,W);
TrainLabels = label(TrainIndex);
TrainingMap(TrainIndex) = TrainLabels;
TrainingMap_map =label2color(TrainingMap,'india');
figure
imshow(TrainingMap_map)
title('Training Set')
axis image;set(gca,'xtick',[],'ytick',[]);

disp('Training Set')
tabulate(TrainingMap(find(TrainingMap>0)))

%% testing set  可视化测试集
TestMap = Label - TrainingMap;
disp('Testing Set')
tabulate(TestMap(find(TestMap>0)))

TestMap_map =label2color(TestMap,'india');
figure
imshow(TestMap_map)
title('Testing Set')
axis image;set(gca,'xtick',[],'ytick',[]);

%% predict
nclasses = length(unique(Label))-1;
[maxp,predict_class] = max(prob_map,[],2);
TestIndex = TestIndex+1;
PredictMap = zeros(H,W);
for i = 1:length(predict_class)
    PredictMap(TestIndex(i)) = predict_class(i);
end
% OA Kappa AA CA
TestLable = label(TestIndex);
[OA_class,kappa_class,AA_class,CA_class] = my_calcError( TestLable, predict_class, 1: nclasses);

PredictMap_map =label2color(PredictMap,'india');
figure
imshow(PredictMap_map)
title(['Predict: ' num2str(OA_class)])
axis image;set(gca,'xtick',[],'ytick',[]);


%% predict
PredictMap =  salinas_gt;
for i = 1:length(predict_class)
    PredictMap(TestIndex(i)) = predict_class(i);
end
PredictMap_map =label2color(PredictMap,'india');
figure
imshow(PredictMap_map)
title(['Visual Result'])
axis image;set(gca,'xtick',[],'ytick',[]);