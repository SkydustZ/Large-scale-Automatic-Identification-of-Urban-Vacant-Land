Codes and data used in the paper 
"Large-scale Automatic Identification of Urban Vacant Land 
Using Semantic Segmentation of High-Resolution Remote Sensing Images".
https://doi.org/10.1016/j.landurbplan.2022.104384

The codes are based on a deep learning approach, using the semantic segmentation network DeepLabv3.

------------------------------
Code and data files:
RS_dataset/             # RAW DATA AND OUTPUTS
    train_raw/              # store all cities' raw images and labels for training 
        -BJ_01_map.jpg          # eg for filename of image: {city}_{ID}_map.jpg
        -BJ_01_label.jpg        # eg for filename of label: {city}_{ID}_label.jpg
    train_input/            # training images and labels of the current city model,
                            # produced by -train.py
    test_raw/               # store all cities' raw images for test and prediction
        -BJ_test01_map.jpg      
    test_output/            # output predicton results, prodeced by -predict.py

data_process/           # DATA PROCESSING
    new_dataset/            # image and label patches, produced by codes
    -cut_img_train.py       # code for splitting train images and labels
    -cut_img_test.py        # code for splitting test images
    -merge_img_test.py      # code for merging test model outputs
    -post_process.py        # code for optimizing model outputs
    -train_val_txt.py       # code for dividing training and validation sets
    -utils.py               # code for generating datasets and defining evaluation metrics

model/                  # DEEP LEARNING MODEL
    best_model/             # store the best deeplabv3 models for prediction
        -DeeplabV3_xxx.pth      # 4 models: SZ, BJ, LZ, and Hybrid, used in the paper
    deeplabv3/              # code and pretrained backbones of deeplabv3
        model/                  # code for deeplabv3 model
            -deeplabv3.py           # main code
            -resnet.py              # encoder
            -aspp.py                # decoder
        pretrained_models/resnet/   # pretrained resnet backbones
            -resnetxxx.pth
    -DeeplabV3_xxx.pth      # the best model in the training process 

log/                    # MODEL TRAINING LOG
    csv/                    # csv files
    -Log.log                # text file

-EDA.py                 # main code for exploring the UVL rate of the training dataset,
                        # which relates to the weighted loss function.
-train.py               # main code for model training
-predict.py             # main code for prediction

-------------------------------
How to run the code:
-Environment:
1. Python 3.7
2. PyTorch torch==1.4.0+cu92

-To train:
1. Put raw images and labels in the folder "RS_dataset/train_raw/" following the filename rules above.
2. Run -train.py for model training. Select city. Adjust args and settings as you like.
3. Find -DeeplabV3_xxx.pth under the folder "model/", which is the best trained model.

-To predict:
1. Put raw images in the folder "RS_dataset/test_raw/" following the filename rules above.
2. Run -predict.py. Select model and city. Adjust args and settings as you like.
3. Find results under the folder "RS_dataset/test_output/".

--------------------------------
Training tips:
1. Use -EDA.py to explore the UVL rate of the training dataset and use [rate, 1-rate] as the weights of the weighted loss function.
2. The spatial resolution of the RS image is recommended to be around 1.6 m. See Appendix B of the paper for explanation. 
3. The default deeplabv3 model in the code uses resnet18 as backbone, which is recommended. See Appendix C for explanation.

--------------------------------
The models and RS_dataset can be found in 
https://www.beijingcitylab.com/projects-1/56-urban-vacancies/
https://pan.baidu.com/s/1zDZ41hTsN2QtGaVFPXvwFA (password: 0qew)