
import glob
from skimage import io
import numpy as np
MAIN_FOLDER = '../dataset/hk_building_age/val/class'
data_files = glob.glob(MAIN_FOLDER + '/*')
class_num = [0,0,0,0,0,0]
for file in data_files:
    label = np.asarray(io.imread(file))
    label = label.astype(np.int64)
    label-=1
    label[label ==0] = 1
    label[label ==-1] = 0
    label-=1
    for i in range(6):
        class_num[i] += np.sum(label==i)
print(class_num/np.sum(class_num))