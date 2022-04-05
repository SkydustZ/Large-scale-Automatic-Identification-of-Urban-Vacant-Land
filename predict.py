#!/usr/bin/env python
# coding: utf-8


import os
# os.environ['OPENCV_IO_MAX_IMAGE_WIDTH']=str(2**64)
# os.environ['OPENCV_IO_MAX_IMAGE_HEIGHT']=str(2**64)
# os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**128)
import cv2

import argparse
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
from matplotlib import pyplot as plt
import shutil

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from model.deeplabv3.model.deeplabv3 import DeepLabV3
from data_process.utils import *
from data_process.cut_img_test import cut_image_test
from data_process.merge_img_test import merge_img_test
from data_process.post_process import visualize_blur_threshold, hybrid


def get_args():
    parser = argparse.ArgumentParser('DL Project')
    parser.add_argument('-m', '--resnet', type=str, default='ResNet18_OS8')
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-l', '--lr', type=float, default=2e-4)
    parser.add_argument('-r', '--resume', action='store_true')
    parser.add_argument('--verbose', type=int, default=2, help='verbose level, > 0 means True')
    parser.add_argument('--fp16', action='store_true', help="whether to use 16-bit float")
    # about fp16: https://zhpmatrix.github.io/2019/07/01/model-mix-precision-acceleration/
    args_ = parser.parse_args()

    return args_

args = get_args()
path = './'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])


def denoise(img, color_thr=200, count_min=20):
    """to exclude background in RS images"""
    def count_color(img, thr=color_thr):
        _, color_num = np.unique(np.mean(np.array(img), axis=2, dtype=np.int16),
                                 return_counts=True)
        count = np.where(color_num > thr, 1, 0).sum()
        return count

    if count_color(img) > count_min:
        return img
    else:
        bg = np.ones_like(np.array(img)) * 255
        bg = Image.fromarray(np.uint8(bg))
        return bg

class RemoteSensingDataset_test(data.Dataset):

    def __init__(self, transforms, test_path):
        self.transforms = transforms
        filenames = os.listdir(test_path)
        self.data_list = [os.path.join(test_path, filename) for filename in filenames]
        print('Read ' + str(len(self.data_list)) + ' images')

    def __getitem__(self, idx):
        img = self.data_list[idx]
        image_fn = os.path.split(self.data_list[idx])[1]
        img = Image.open(img)
        img = denoise(img)
        img = self.transforms(img)
        return img, image_fn

    def __len__(self):
        return len(self.data_list)


def predict_city(city, model_path, r=20, thr=0.8):

    def visualize_preds_confidence(preds, image_fns, pred_path):
        """ preds: [bs,w,h]; image_fns: file names"""

        for i in range(len(preds)):
            pred, image_fn = preds[i], image_fns[i]
            pred_fn = image_fn.replace('map', 'pred')

            pred = (1 - pred.detach().cpu().numpy()) * 255
            pred = Image.fromarray(np.uint8(pred))

            pred.save(pred_path + pred_fn)

    def predict():
        test_data = tqdm(test_data_loader, position=0, leave=True)
        net.eval()
        with torch.no_grad():
            for data in test_data:
                images = data[0].cuda()
                image_fns = data[1]
                outputs = net(images)  # [bs,2,w,h]
                preds_conf = F.softmax(outputs, dim=1)[:, 1, :, :]

                visualize_preds_confidence(preds_conf, image_fns, pred_path)

    cut_image_test(city)

    test_path = f'data_process/new_dataset/test_image_{city}/'
    test_dataset = RemoteSensingDataset_test(transform, test_path)
    test_data_loader = data.DataLoader(test_dataset, args.batch_size)

    pred_path = f'data_process/new_dataset/test_pred_{city}/'
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)
    else:
        shutil.rmtree(pred_path)
        os.mkdir(pred_path)

    net = DeepLabV3(args.resnet, path)
    net = net.to(device)
    net.load_state_dict(torch.load(model_path))
    predict()

    merge_img_test(city)

    # visualize_blur_threshold
    file_dir = 'RS_dataset/test_output/'
    visualize_blur_threshold(file_dir, city, r=r, thr=thr)


# -----------------------------------------------------
if __name__=='__main__':
    # southern, SZ model
    city_list1 = [
        ('CD','成都'),  
        ('CQ','重庆'),  
        ('CS','长沙'),  
        ('FZ','福州'),  
        ('GY','贵阳'), 
        ('GZ','广州'),  
        ('HF','合肥'), 
        ('HK','海口'), 
        ('HZ','杭州'), 
        ('KM','昆明'),
        ('NB','宁波'), 
        ('NC','南昌'),
        ('NJ','南京'),  
        ('NN','南宁'),
        ('SH','上海'), 
        ('SZ','深圳'),
        ('XM','厦门'), 
        ('WH','武汉'),
        ]

    # northern-A, BJ model
    city_list2 = [
        ('BJ', '北京'), 
        ('TJ', '天津'),
        ('ZZ', '郑州'),
        ('TY', '太原'),
        ('XA', '西安'),
        ('QD', '青岛'),
        ('JN', '济南'), 
        ]

    # northern-B, LZ model
    city_list3 = [
        ('SJ', '石家庄'),
        ('CC', '长春'),
        ('DL', '大连'), 
        ('SY', '沈阳'),
        ('HE', '哈尔滨'), 
        ('LZ', '兰州'), 
        ('HH', '呼和浩特'),
        ('YC', '银川'),
        ('WL', '乌鲁木齐'),
        ('LS', '拉萨'),
        ('XN', '西宁'),
        ]

    Image.MAX_IMAGE_PIXELS = int(5000000000)

    # # set model path for prediction
    # model_path = 'model/best_model/DeeplabV3_ResNet18_OS8_SZ -lr 0.0002 -bs 32 -f2 0.852 -iou 0.649 -loss 0.210 -ep 19.pth'
    model_path = 'model/best_model/DeeplabV3_ResNet18_OS8_BJ -lr 0.0002 -bs 32 -f2 0.888 -iou 0.719 -loss 0.167 -ep 29.pth'
    # model_path = 'model/best_model/DeeplabV3_ResNet18_OS8_LZ -lr 0.0002 -bs 32 -f2 0.817 -iou 0.550 -loss 0.218 -ep 24.pth'
    # model_path = 'model/best_model/DeeplabV3_ResNet18_OS8_Hybrid -lr 0.0002 -bs 32 -f2 0.841 -iou 0.613 -loss 0.211 -ep 17.pth'

    city = 'BJ'
    predict_city(city, model_path, r=20, thr=0.6)


    # # use hybrid prediction if needed
    # file_dir = 'RS_dataset/test_output/'
    # filepath1 = file_dir+'CD_test01SZ_pred.jpg'
    # filepath2 = file_dir+'CD_test01LZ_pred.jpg'
    # hybrid(filepath1, filepath2, r=20, thr=0.5, ratio=(1.0,1.0))

