# -*- coding:utf-8 -*-
import time     
import os  
import shutil
import random
 
 
def readFilename(path):
    allfile = []
    filelist = os.listdir(path)
 
    for filename in filelist:
        filepath = os.path.join(path, filename)
        allfile.append(filepath)
        
    return allfile
 
def train_val_txt(path='./'):
    path=path+"data_process/"
    img_path=path+f"new_dataset/image/"
    allfile=readFilename(img_path)
    random.shuffle(allfile)  
    
    # allname=[]
    train_txtpath = path+f"new_dataset/train.txt"
    val_txtpath = path+f"new_dataset/val.txt"

    picturenum = len(allfile)
    train_num = int(0.75*picturenum)
    train_content=''
    test_content=''

    for i,name in enumerate(allfile):
        file_name = name.split("/")[-1].split(".")[0]
        # print(file_name)
        if i < train_num:
            train_content += (file_name + ' \n')
        else:
            test_content += (file_name + '\n')

    with open(train_txtpath, 'w') as tfp:
        tfp.write(train_content + "\n")
    with open(val_txtpath, 'w') as vfp:
        vfp.write(test_content + "\n")
 
if __name__ == '__main__':
    path = './'
    train_val_txt(path=path)