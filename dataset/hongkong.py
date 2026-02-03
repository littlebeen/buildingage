import torch
import glob
import random
import os
import numpy as np
from skimage import io

def generate_first_impervious_year(imperv_data):
    # 步骤1：数据维度校验与标准化（确保输入为4×W×H，类型为np.ndarray）
    n_times, H, W = imperv_data.shape
    assert n_times == 4, "输入数据应包含4个时相的impervious信息"
    imperv = imperv_data.copy() 

    # 步骤2：定义4个时相对应的年份（严格按1990/2000/2010/2020顺序）
    year_mapping = np.array([3, 4, 5, 6])
    first_imperv_map = np.full(shape=(1, H, W), fill_value=-1, dtype=np.int32)

    # 步骤3：创建有效像素掩码（排除缺测值0，仅保留1/2的有效像素）
    valid_mask = (imperv != 0)
    valid_pixel = valid_mask.all(axis=0)  # 形状(H, W)，True=该像素4个时相均无缺测

    imperv_mask = (imperv == 2)  # 形状(4, H, W)，True=不透水面
    first_imperv_idx = np.argmax(imperv_mask, axis=0)  # 形状(H, W)，值为0/1/2/3（对应4个时相）
    # 修正：无不透水面的有效像素，首次索引设为-1
    no_imperv = ~imperv_mask.any(axis=0)  # 形状(H, W)，True=全程无不透水面
    first_imperv_idx[no_imperv] = -1

    # 步骤5：验证首次不透化后是否永久保持（排除1→2→1的多次突变）
    permanent_imperv = np.zeros(shape=(H, W), dtype=bool)  # 形状(H, W)，True=永久不透化
    for i in range(H):
        for j in range(W):
            idx = first_imperv_idx[i, j]
            if idx == -1:
                continue  # 全程无不透水面，跳过
            if valid_pixel[i, j]:
                # 检查首次不透化后所有时相是否均为不透水面（2）
                if imperv_mask[idx:, i, j].all():
                    permanent_imperv[i, j] = True

    ## 情况2：首次永久不透化的像素 → 赋值对应年份（1990/2000/2010/2020）
    for idx in range(4):
        # 找到该时相首次永久不透化的像素
        target_pixel = (first_imperv_idx == idx) & permanent_imperv & valid_pixel
        first_imperv_map[0, target_pixel] = year_mapping[idx]

    ## 情况3：有效像素中全程透水（无不透水面）或多次突变（非永久不透化）→ 0（无建筑背景）
    background_pixel = (no_imperv | ~permanent_imperv) & valid_pixel  # 形状(H, W)
    first_imperv_map[0, background_pixel] = -1
    return first_imperv_map


def get_year_type(arr_processed):
    # arr_processed[(arr_processed >8) & (arr_processed <25)] = 1
    # arr_processed[(arr_processed >35) & (arr_processed <55)] = 2
    # arr_processed[arr_processed > 55] = 3
    arr_processed-=1
    arr_processed[arr_processed ==0] = 1
    arr_processed[arr_processed ==-1] = 0
    return arr_processed

def get_ufz_type(arr_processed):
    arr_processed[(arr_processed >= 1) & (arr_processed <= 4)] = 1
    arr_processed[(arr_processed >= 5) & (arr_processed <= 9)] = 2
    return arr_processed

class Hongkong_dataset(torch.utils.data.Dataset):
    def __init__(self, mode,cache=False, augmentation=True):
        super(Hongkong_dataset, self).__init__()
        MAIN_FOLDER = '../dataset/hk_building_age/'+mode+'/'
        DATA_FOLDER = MAIN_FOLDER + 'image/tdop*.tif'
        self.LABEL_FOLDER = MAIN_FOLDER + 'class/'
        self.BOUNDARY_FOLDER = MAIN_FOLDER + 'mask/'
        self.HEIGHT_FOLDER = MAIN_FOLDER + 'height/'
        self.UFZ_FOLDER = MAIN_FOLDER + 'ufz/'
        self.mode = mode
        self.augmentation = augmentation
        self.cache = cache
        self.max_num=0

        # List of files
        self.data_files = glob.glob(DATA_FOLDER)
        # self.boundary_files = glob.glob(BOUNDARY_FOLDER)
        # self.height_files = glob.glob(HEIGHT_FOLDER)
        # self.label_files = glob.glob(LABEL_FOLDER)

        # Sanity check : raise an error if some files do not exist
        # for f in self.data_files + self.label_files:
        #     if not os.path.isfile(f):
        #         raise KeyError('{} is not a file !'.format(f))


    def __len__(self):
        # Default epoch size is 10 000 samples
        return len(self.data_files)

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        name=self.data_files[i].split('/')[-1].split('.')[0].replace('image', '')
        data = io.imread(self.data_files[i])[:, :, :3].transpose((2, 0, 1))
        data = 1 / 255 * np.asarray(data, dtype='float32')

        height_files = self.HEIGHT_FOLDER+name+'height.tif'
        height = io.imread(height_files)
        height = np.asarray(height, dtype='float32')
        height = height - height.min()
        height = height / 500.0  # normalize to 0-1

        ufzs=[]
        for year in ['1990','2000','2010','2020']:
            ufz = io.imread(self.UFZ_FOLDER+name+'ufz_'+year+'.tif')
            ufz = np.asarray(ufz, dtype=np.float32)
            ufz=get_ufz_type(ufz)
            # unique_values1 = np.unique(ufz)
            # print(unique_values1)
            ufzs.append(ufz)

        boundary_files = self.BOUNDARY_FOLDER+name+'mask.png'
        boundary = np.asarray(io.imread(boundary_files)) / 255
        boundary = boundary.astype(np.float32)

        label = np.asarray(io.imread(self.LABEL_FOLDER+name+'class.png'))
        label = label.astype(np.int64)

        label = get_year_type(label)
        # unique_values1 = np.unique(label)
        # print(unique_values1)

        # Data augmentation
        # data_p, boundary_p, label_p = self.data_augmentation(data_p, boundary_p, label_p)
        if self.mode == 'train' and self.augmentation:
            data, boundary, height, label, ufzs[0],ufzs[1], ufzs[2], ufzs[3] = self.data_augmentation(data, boundary, height, label,ufzs[0],ufzs[1], ufzs[2], ufzs[3])

        height = height[np.newaxis, :, :]
        boundary = boundary[np.newaxis, :, :]
        # height = np.repeat(height, repeats=3, axis=0)
        ufzs = np.stack(ufzs, axis=0)
        ufzs = generate_first_impervious_year(ufzs)
        # unique_values1 = np.unique(ufzs)
        # print(unique_values1)
        return (torch.from_numpy(data),
                torch.from_numpy(boundary),
                torch.from_numpy(height),
                torch.from_numpy(ufzs),
                torch.from_numpy(label)-1)