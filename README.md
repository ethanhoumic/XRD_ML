# Machine Learning in Physics Final Project 

## Topic
Using machine learning on XRD analysis

## Introduction and Expected Achievement

In this project, we have used various model to predict the crystal system from XRD spectrum. We end up achieved a highest accuracy of 67 % to classify the 7 crystal system from XRD data.
  
## Members (Name and Github Username) 

1. 侯奕安
2. 吳政蔚 (wasslu914)
3. 蔡杰達 (tsaijieda)

## Method

### Data Preparation
We first obtain numerous Crystal Information File from Crystallography Open Database. We then remove the data that are organic or having big crystal structure, then feed these file through python package PyMatgen to give us the XRD peak postition and intensity. Finally, we did Gaussian convolution on the peak data and add some random noise to simulate experimental XRD data. The processed data is split as the follwing table

| crystal system | train | test |
|----------------|-------|------|
| tetragonal     | 3000  | 500  |
| orthorhombic   | 3500  | 500  |
| trigonal       | 2000  | 500  |
| cubic          | 2000  | 500  |
| triclinic      | 1800  | 500  |
| monoclinic     | 3200  | 500  |
| hexagonal      | 100   | 10   |

### Training
We've trained five different model, including Dense, CNN + Dense, Extreme Randomized Trees, CNN + LSTM and Transformer. The training details are presented in the slides. The final accuracy result is shown in the following table

| Model                           | Full spectrum (%) | Peaks only (%) |
|---------------------------------|-------------------|----------------|
| Dense (150 MB)                  | 35                | 56             |
| CNN+Dense (160 MB)              |                   | 60             |
| CNN-LSTM with CE/FL (1 MB)      |                   | 64/54          |
| sCNN-LSTM with CE/FL (75 KB)    | 52/50             | 43/42          |
| Extreme Randomized Tree (90 MB) |                   | 67             |
| Transformer (460 KB)            | 33                | 60             |

## Planning and Timeline

| Week   | Task |
|--------|------|
| 11/12  | Topic brainstorm     |
| 11/19  | Data preparation     |
| 11/26  | Training model     |
| 12/3  |  Training model    |

侯奕安 will be responsible for training 4 models including CNN-LSTM with CE/FL and sCNN-LSTM with CE/FL.

吳政蔚 will be responsible for training 1 model, Transformer.

蔡杰達 will be responsible for data preparation and training 3 models including Dense, CNN + Dense and Extreme Randomized Tree.

## User Guide

### Data and Specification

The data are stored on Google Drive https://drive.google.com/drive/folders/15rCZyiqVFdUs5Vrj4TXOc6qKFB-aUseI?usp=share_link , and the code is designed to run on Colab using GPU T4. 

吳政蔚's code uses local files (I downloaded all the data), so that part of the code has to be rewritten if one wants to run it on colab. 

For 侯奕安's notebooks (model_CNNLSTM.ipynb, model_CNN_LSTM_peak.ipynb), please follow the below hierarchy to execute the code: 

|-- your google drive <br>
-|-- xrd_training  <br>
--|-- structure_info.csv, output_data.zip <br>
For 蔡杰達's .pt file, because most of them have big size, they are stored in the same Google Drive link above.

### Model 
#### Dense, CNN + Dense, Extra Trees
The models can be executed in the notebook file on Github, run the code from the first section to last.

#### CNN-BiLSTM
The models can be trained and evaluated by following the cells. If you want to directly evaluate an existing .pt file, add a cell with "model = torch.load("model_path")". "CE" in the weight file name means that the loss function is CrossEntropy, while "FL" means the loss function is FocalLoss. 

## References
[1] Vecsei, P. M., Choo, K., Chang, J., & Neupert, T. (2019). Neural network based classification of crystal symmetries from x-ray diffraction patterns. Physical Review. B./Physical Review. B, 99(24). https://doi.org/10.1103/physrevb.99.245120

[2] Suzuki, Y., Hino, H., Hawai, T., Saito, K., Kotsugi, M., & Ono, K. (2020b). Symmetry prediction and knowledge discovery from X-ray diffraction patterns using an interpretable machine learning approach. Scientific Reports, 10(1). https://doi.org/10.1038/s41598-020-77474-4
