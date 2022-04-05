# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np
import cv2
import shutil

def splitimage_nogap(img_name, new_img_path, patch_size):
    img = Image.open(img_name)

    w, h = img.size
    w_new = (w // patch_size) * patch_size
    h_new = (h // patch_size) * patch_size

    s_img = os.path.split(img_name)
    fn_img = s_img[1].split('.')
    basename_img = fn_img[0]

    num = 1
    rowheight = patch_size
    colwidth = patch_size
    for r in range(h_new // patch_size+1):
        for c in range(w_new // patch_size+1):
            if c < w_new // patch_size and r < h_new // patch_size:
                box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)


            elif c == w_new // patch_size and r < h_new // patch_size:
                box = (w - colwidth, r * rowheight, w, (r + 1) * rowheight)

            elif c < w_new // patch_size and r == h_new // patch_size:
                box = (c * colwidth, h-rowheight, (c + 1) * colwidth, h)

            else:
                box = (w - colwidth, h-rowheight, w, h)

            img_new_path = os.path.join(new_img_path, basename_img + '_' + str(num) + '.jpg')
            img.crop(box).save(img_new_path, quality=95, subsampling=0)
            # print(img_new_path)
            
            # label_color_new_path = os.path.join(new_path, basename_img + '_' + str(num) + '_color.png')
            # label_color.crop(box).save(label_color_new_path)
            num = num + 1
    # print('--------------------------------------------------------------------------------')


def cut_image_test(city, path='./', patch_size=224):

    source_path = path+'RS_dataset'
    test_path = os.path.join(source_path, 'test_raw')
    new_img_path = path+f'data_process/new_dataset/test_image_{city}/'
    if not os.path.exists(new_img_path):
        os.mkdir(new_img_path)
    else:
        shutil.rmtree(new_img_path)
        os.mkdir(new_img_path)

    # #split
    filename = os.listdir(test_path)
    for name in filename:
        if name.find(city) >= 0:
            if name.find('class') >= 0 or name.find('label') >= 0:
                continue
            else:
                temp_img_path = os.path.join(test_path, name)
                splitimage_nogap(temp_img_path, new_img_path, patch_size)

Image.MAX_IMAGE_PIXELS = int(5000000000)

if __name__=='__main__':
    #split for no gap
    path = '../'
    city = 'CD'
    cut_image_test(city, path=path)