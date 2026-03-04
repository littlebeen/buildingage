import shutil
target_forder='/mnt/d/Jialu/dataset/hk_building_age/train/image/'
text_path='./train.txt'

with open(text_path, "r", encoding="utf-8") as f:
        data_files = [line.strip() for line in f if line.strip()]
for file in data_files:
    shutil.move(file, target_forder+file.split('/')[-1])
