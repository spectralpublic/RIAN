# RIAN

# Rotation-Invariant Attention Network for Hyperspectral Image Classification


### 1. Introduction

This is the reserch code of the IEEE Transactions on Image Processing 2022 paper, which is currently under review.

X. Zheng, H. Sun, X. Lu, and W. Xie, “Rotation-Invariant Attention Network for Hyperspectral Image Classification,” IEEE Transactions on 
Image Processing, 2022.

The details of files and directories in this repository are shown as follows.

| Item                 			| Description	|
|---------------------------------------|------------------------------------------------------------------|
| ./Data                 		| The directory containing hyperspectral images.|
| ./result_RIAN				| The directory containing hyperspectral images.|
| attention.py 				| Code of the proposed CSpeA and RSpaA modules.|
| HSI_Data_Preparation_Houston.py	| Code for splitting Houston2013 data set into training set and testing set.|
| HSI_Data_Preparation_PU.py		| Code for splitting Pavia University data set into training set and testing set.|
| HSI_Data_Preparation_Salinas.py	| Code for splitting Salinas data set into training set and testing set.|
| RIAN.py 				| The main manuscript of the proposed RIAN.|
| utils_houston.py			| Parameter settings for Houston2013 data set.|
| utils_PU.py				| Parameter settings for Pavia University data set.|
| utils_salinas.py			| Parameter settings for Salinas data set.|

### 2. Start


Requirements:
             
	Python 3
	
	tensorflow-1.4.1

	scikit-image

1. Split training set and testing set. 

Three data sets are used: Houston_2013，Pavia University and Salinas, which can be downloaded from Google Drive https://drive.google.com/file/d/1BkwgTRh3JtKqQcLQ7DkMv98jChs15JmB/view?usp=sharing.

Run "HSI_Data_Preparation_Houston.py", "HSI_Data_Preparation_PU.py" and "HSI_Data_Preparation_Salinas.py " to get training set and testing set of each data set.


2. Run "python RIAN.py" for training and testing.

Revise lines 27-33 to change data set. Testing results are saved in the directory "result_RIAN".


3. Run "result_RIAN/DATASETNAME/acc_in_testing_set.m" in Matlab to calculate evaluation metrics such as OA, AA and kappa. 

"DATASETNAME" is the name of data set.

4. For the codes of compared methods in the paper, please refer to:

DHCNet: https://github.com/ordinarycore/DHCNet

SSRN: https://github.com/zilongzhong/SSRN

1-D, 2D-CNN：https://github.com/nshaud/DeepHyperX

SMBN, SFFN：http://www.escience.cn/people/LeyuanFang/index.html

MGCN: https://github.com/danfenghong/IEEE_TGRS_GCN



### 3. Related work

The paper of this code is currently under review. If you find the code and dataset useful in your research, please consider citing our paper.


X. Zheng, H. Sun, X. Lu, and W. Xie, “Rotation-Invariant Attention Network for Hyperspectral Image Classification,” IEEE Transactions on 
Image Processing, 2022.

