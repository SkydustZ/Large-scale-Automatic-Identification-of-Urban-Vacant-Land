# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np
import cv2
import shutil
from .train_val_txt import train_val_txt
      
def trim(img_map, img_label, type= 'white'):
    # trim type

    img1 = img_map.sum(axis=2)
    (row, col) = img1.shape
    tempr0 = 0
    tempr1 = 0
    tempc0 = 0
    tempc1 = 0
    # 765 是255+255+255,如果是黑色背景就是0+0+0，彩色的背景，将765替换成其他颜色的RGB之和，这个会有一点问题，因为三个和相同但颜色不一定同
    for r in range(0, row):
        if type == 'white':
            if img1.sum(axis=1)[r] <= 750 * col:
                tempr0 = r
                break
        else:
            if img1.sum(axis=1)[r] >= 3 * col:
                tempr0 = r
                break

    for r in range(row - 1, 0, -1):
        if type == 'white':
            if img1.sum(axis=1)[r] <= 750 * col:
                tempr1 = r
                break
        else:
            if img1.sum(axis=1)[r] >= 3 * col:
                tempr1 = r
                break

    for c in range(0, col):
        if type == 'white':
            if img1.sum(axis=0)[c] <= 750 * row:
                tempc0 = c
                break
        else:
            if img1.sum(axis=0)[c] >= 3 * row:
                tempc0 = c
                break

    for c in range(col - 1, 0, -1):
        if type == 'white':
            if img1.sum(axis=0)[c] <= 750 * row:
                tempc1 = c
                break
        else:
            if img1.sum(axis=0)[c] >= 3 * row:
                tempc1 = c
                break

    new_img_map = img_map[tempr0:tempr1 + 1, tempc0:tempc1 + 1, 0:3]
    new_img_label = img_label[tempr0:tempr1 + 1, tempc0:tempc1 + 1, 0:3]
    new_img_class = np.where(new_img_label.sum(axis=2)<600,255,0)  # white 0(not space land), others 1(space land)

    return new_img_map, new_img_label, new_img_class

def get_img_class(city, path='./'):
    source_path = path+'RS_dataset'
    rawdata_path = os.path.join(source_path, 'train_raw')

    def filename(file_dir):
        maps = []
        labels = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':
                    if (city == 'Hybrid') or (city in file):
                        if os.path.splitext(file)[0][-3:] == 'map':
                            maps.append(os.path.join(root, file))
                        else:
                            labels.append(os.path.join(root, file))
        return maps, labels

    maps, labels= filename(rawdata_path)
    for i in range(len(maps)):
        mapdir = maps[i]
        labeldir = mapdir.replace('map','label')

        img_map = cv2.imread(mapdir)
        img_label = cv2.imread(labeldir)
        new_img_map, new_img_label, new_img_class = trim(img_map, img_label)

        train_mapdir = mapdir.replace('train_raw', f'train_input')
        train_labeldir = labeldir.replace('train_raw', f'train_input')
        train_classdir = train_labeldir.replace('label', 'class')
        cv2.imwrite(train_mapdir, new_img_map, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(train_labeldir, new_img_label, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(train_classdir, new_img_class, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print('Image No.'+str(i+1)+' has been processed')

def img_label(img_path):
    label_name = img_path.replace('map', 'class')
    color_name = img_path.replace('map', 'label')
    return label_name, color_name

def splitimage(img_name, new_img_path, new_label_path, patch_size, stride):
    img = Image.open(img_name)
    label_name, label_color_name = img_label(img_name)
    label = Image.open(label_name)
    label_color = Image.open(label_color_name)

    w, h = img.size
    w_new = ((w-patch_size) // stride) * stride + patch_size
    h_new = ((h-patch_size) // stride) * stride + patch_size
    # img_new = img.crop((0, 0, w_new, h_new))
    # label_new = label.crop((0, 0, w_new, h_new))
    # label_color_new = label_color.crop((0, 0, w_new, h_new))
    
    s_img = os.path.split(img_name)
    fn_img = s_img[1].split('.')
    basename_img = fn_img[0]

    num = 1
    rowheight = stride
    colwidth = stride
    for r in range((h_new-patch_size) // stride + 1):
        for c in range((w_new-patch_size) // stride + 1):
            x = c * colwidth
            y = r * rowheight
            box = (x, y, x+patch_size, y+patch_size)
            
            img_new_path = os.path.join(new_img_path, basename_img + '_' + str(num) + '.jpg')
            img.crop(box).save(img_new_path, quality=95, subsampling=0)
            
            label_new_path = os.path.join(new_label_path, basename_img + '_' + str(num) + '.jpg')
            label.crop(box).save(label_new_path, quality=95, subsampling=0)
            
            # label_color_new_path = os.path.join(new_path, basename_img + '_' + str(num) + '_color.png')
            # label_color.crop(box).save(label_color_new_path)
            
            # print(img_new_path)
            num = num + 1
    # print('--------------------------------------------------------------------------------')


def cut_image_train(city, path='./', patch_size=256, stride=224):
    """city in ["SZ", "BJ", "LZ", "Hybrid"]"""
    source_path = path+'RS_dataset'
    new_img_path = path+f'data_process/new_dataset/image/'
    new_label_path = path+f'data_process/new_dataset/label/'
    train_path = os.path.join(source_path, f'train_input')

    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(new_img_path):
        shutil.rmtree(new_img_path)
        shutil.rmtree(new_label_path)

    os.mkdir(train_path)
    os.mkdir(new_img_path)
    os.mkdir(new_label_path)

    # rawdata process
    get_img_class(city, path)

    # #split
    filename = os.listdir(train_path)
    for name in filename:
        if name.find('class') >= 0 or name.find('label') >= 0:
            continue
        else:
            temp_img_path = os.path.join(train_path, name)
            splitimage(temp_img_path, new_img_path, new_label_path, patch_size=patch_size, stride=stride)

    train_val_txt(path=path)

Image.MAX_IMAGE_PIXELS = int(5000000000)

if __name__=='__main__':

    path = '../'
    city = 'Hybrid'
    cut_image_train(city, path=path, patch_size=256, stride=224)
