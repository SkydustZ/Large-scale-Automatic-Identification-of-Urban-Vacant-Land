import os
os.environ['OPENCV_IO_MAX_IMAGE_WIDTH']=str(2**64)
os.environ['OPENCV_IO_MAX_IMAGE_HEIGHT']=str(2**64)
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**128)
import cv2

from PIL import Image
import numpy as np

def blur(filepath, r=50, mode='pred'):
    # r=50 is the best
    raw_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    blur_img = cv2.blur(raw_img, (r,r))
    to_path = filepath.replace(mode, f'blur{r}')
    cv2.imwrite(to_path, blur_img)
    return to_path

def image_threshold(filepath, thr=0.7):
    # thr=0.7 is the best
    img = Image.open(filepath)
    img = np.array(img)
    thr2 = (1-thr) * 255
    img = np.where(img > thr2, 255, 0)
    img = Image.fromarray(np.uint8(img))
    return img

def visualize_blur_threshold(file_dir, city, r=20, thr=0.6, mode='pred'):
    l = len(mode)
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.find(city) >= 0:
                if os.path.splitext(file)[0][-l:] == mode:
                    filepath = os.path.join(root, file)
                    blur_path = blur(filepath, r=r, mode=mode)

                    # to_path = blur_path.replace(f'blur{r}', f'label')
                    to_path = blur_path.replace(f'blur{r}', f'label').replace('jpg', 'tif')
                    img = image_threshold(blur_path, thr=thr)
                    img.save(to_path)
                    os.remove(blur_path)

def hybrid(filepath1, filepath2, r=20, thr=0.5, ratio=(1.0, 1.0)):
    img1 = np.array(Image.open(filepath1))
    img2 = np.array(Image.open(filepath2))
    img = (img1 * ratio[0] + img2 * ratio[1]) / (ratio[0] + ratio[1])
    img = Image.fromarray(np.uint8(img))
    to_path = filepath1.replace('_pred', '_hybrid')
    img.save(to_path)
    
    blur_path = blur(to_path, r=r, mode='hybrid')
    img = image_threshold(blur_path, thr=thr)

    img.save(to_path)
    to_path = to_path.replace('jpg', 'tif')
    img.save(to_path)
    os.remove(blur_path)

if __name__=='__main__':
    Image.MAX_IMAGE_PIXELS = int(5000000000)
    file_dir = '../RS_dataset/test_output/'
    # city = 'NJ'
    # visualize_blur_threshold(file_dir, city, r=20, thr=0.6)

    filepath1 = file_dir+'CD_test01SZ_pred.jpg'
    filepath2 = file_dir+'CD_test01LZ_pred.jpg'
    hybrid(filepath1, filepath2, r=20, thr=0.5, ratio=(1.0,1.0))