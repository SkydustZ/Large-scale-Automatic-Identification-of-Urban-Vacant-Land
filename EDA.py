from data_process.utils import *
from data_process.cut_img_train import cut_image_train

path = './'
city = 'BJ'
# cut_image_train(city, path=path, patch_size=256, stride=224)

train = True
dataset = RemoteSensingDataset(train, transform_val, path)
total = len(dataset)

label_pixel = 224**2
count_pixel = 0
total_pixel = 0

for i in range(total):
    label = dataset[i][1]
    count_pixel += label.sum().item()
    total_pixel += label_pixel
    
pixel_ratio = count_pixel / total_pixel
print(f"UVL_rate = {pixel_ratio}")
