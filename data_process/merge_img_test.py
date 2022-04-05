import os
import math
from PIL import Image
import numpy as np
import cv2
import pandas as pd

def merge(pre_path:'str', origin_map_path, save_path, splitstr='_map_', patch_size=224):
    def get_filepath(file_dir):
        maps = []
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':
                        maps.append(os.path.join(root, file))
        return maps
    def get_mergefilename(filepaths):
        images={}
        for filepath in filepaths:
            filename=os.path.basename(filepath)
            imagename, image_subnumber = filename.split(splitstr)
            image_subnumber = int(image_subnumber[:-4])
            if imagename not in images:
                images[imagename] = [(filepath,image_subnumber)]
            else:
                images[imagename].append((filepath,image_subnumber))
        for image in images:
            images[image]= sorted(images[image], key=lambda x:x[1])
        return images
    
    def get_orimap_size(imagename):
        oripath=os.path.join(origin_map_path,imagename+'_map.jpg')
        oriimage = Image.open(oripath)
        return oriimage.size

    def image_merge_nogap(mergefilename, image_size=224, savepath='./new_dataset/merge_nogap'):
        number = 1
        for onefilename in mergefilename:
            picture_dirs = mergefilename[onefilename]

            # get to_image name
            filedir = picture_dirs[0][0]
            filename = os.path.basename(filedir)
            imagename = filename.split(splitstr)[0]
            w, h =get_orimap_size(imagename)

            image_col = w // image_size + 1
            image_row = h // image_size + 1

            to_image = Image.new('RGB', (w, h))
            to_image_path = os.path.join(savepath, imagename + splitstr[:-1] + '.jpg')

            for i in range(len(picture_dirs)):
                picture_dir = picture_dirs[i][0]
                picture_number = picture_dirs[i][1]
                x = i // image_col # row
                y = i - x * image_col # col
                if (x+1) % image_row != 0 and (y+1) % image_col !=0:
                    point1 = y * image_size # x-direction
                    point2 = x * image_size # y-direction
                elif (x+1) % image_row == 0 and (y+1) % image_col !=0:
                    point1 = y * image_size
                    point2 = h - image_size
                elif (x+1) % image_row != 0 and (y+1) % image_col ==0:
                    point1 = w - image_size
                    point2 = x * image_size
                else:
                    point1 = w - image_size
                    point2 = h - image_size
                from_image = Image.open(picture_dir)
                # print("x "+str(x)+'; y '+str(y))
                to_image.paste(from_image, (point1, point2,point1+image_size,point2+image_size))

            to_image.save(to_image_path, quality=95, subsampling=0)
            print('Image No.' + str(number) + ' has been processed')
            number += 1
    
    filepaths=get_filepath(pre_path)
    mergefilename = get_mergefilename(filepaths)
    image_merge_nogap(mergefilename, savepath=save_path, image_size=patch_size)


def merge_img_test(city, path='./', patch_size=224):
    origin_map_path = path+'RS_dataset/test_raw'
    savepath = path+'RS_dataset/test_output'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    # merge map test
    prepath = path+f'data_process/new_dataset/test_image_{city}'
    splitstr='_map_'
    merge(prepath, origin_map_path, savepath, splitstr=splitstr, patch_size=patch_size)

    # merge pred test
    prepath = path+f'data_process/new_dataset/test_pred_{city}'
    splitstr='_pred_'
    merge(prepath, origin_map_path, savepath, splitstr=splitstr, patch_size=patch_size)

Image.MAX_IMAGE_PIXELS = int(2800000000)

if __name__ =="__main__":
    path = '../'
    city = 'XA'
    merge_img_test(city, path=path)

