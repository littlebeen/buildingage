import shutil
import os
import glob
target_forder='/mnt/d/Jialu/dataset/hk_building_age/train/ufz/'
text_path='./test.txt'

# with open(text_path, "r", encoding="utf-8") as f:
#         data_files = [line.strip().replace('image','class').replace('.tif','.png') for line in f if line.strip()]
MAIN_FOLDER = '../dataset/hk_building_age/train/image'
data_files = glob.glob(MAIN_FOLDER + '/*')
for file in data_files:
    name =file.replace('image','ufz').replace('.tif','').replace('train','val')
    DATA_FOLDER = name + '*.tif'
    data_files = glob.glob(DATA_FOLDER)
    for tif_file in data_files:
        shutil.move(tif_file, target_forder+name.split('/')[-1]+'.tif')
#      os.remove(file)
