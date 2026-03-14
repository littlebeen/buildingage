import torch
import glob
import random
import os
import numpy as np
from skimage import io
from scipy.ndimage import label as label11
from config import convert_to_color,save_img
from scipy.stats import mode
import albumentations as A

def generate_instance_mask(mask,min_count_threshold=200):
    mask_flat = mask.flatten()
    unique_ids, counts = np.unique(mask_flat, return_counts=True)
    
    # Step 2: 构建「原ID→出现次数」字典（排除0，0默认是背景）
    id_count_dict = dict(zip(unique_ids, counts))
    
    # Step 3: 筛选保留的实例（次数≥阈值，且原ID≠0）
    retained_ids = [
        id_ for id_ in unique_ids 
        if id_ != 0 and id_count_dict[id_] >= min_count_threshold
    ]
    
    # Step 4: 构建重编码映射表（原ID→新ID，从1开始连续编号）
    id_mapping = {0: 0}  # 背景0保持不变
    for new_id, old_id in enumerate(retained_ids, start=1):
        id_mapping[old_id] = new_id
    
    # Step 5: 低频实例归为0（未出现在retained_ids中的非0ID）
    for old_id in unique_ids:
        if old_id != 0 and old_id not in retained_ids:
            id_mapping[old_id] = 0
    
    # Step 6: 应用映射表，生成重编码后的mask
    # 用np.vectorize高效替换值（支持任意维度）
    vectorized_mapping = np.vectorize(lambda x: id_mapping[x])
    reencoded_mask = vectorized_mapping(mask)
    
    
    return reencoded_mask, len(retained_ids)


def generate_first_impervious_year(imperv_data):
    # 步骤1：数据维度校验与标准化（确保输入为4×W×H，类型为np.ndarray）
    n_times, H, W = imperv_data.shape
    assert n_times == 4, "输入数据应包含4个时相的impervious信息"
    imperv = imperv_data.copy() 

    # 步骤2：定义4个时相对应的年份（严格按1990/2000/2010/2020顺序）
    year_mapping = np.array([3, 4, 5, 6])
    first_imperv_map = np.full(shape=(1, H, W), fill_value=0, dtype=np.int32)

    # 步骤3：创建有效像素掩码（排除缺测值0，仅保留1/2的有效像素）
    valid_mask = (imperv != 0)
    valid_pixel = valid_mask.all(axis=0)  # 形状(H, W)，True=该像素4个时相均无缺测

    imperv_mask = (imperv == 2)  # 形状(4, H, W)，True=不透水面
    first_imperv_idx = np.argmax(imperv_mask, axis=0)  # 形状(H, W)，值为0/1/2/3（对应4个时相）
    # 修正：无不透水面的有效像素，首次索引设为-1
    no_imperv = ~imperv_mask.any(axis=0)  # 形状(H, W)，True=全程无不透水面
    first_imperv_idx[no_imperv] = 0

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
    first_imperv_map[0, background_pixel] = 0
    return first_imperv_map





def get_year_type(label):
    arr_processed=label.copy()
    arr_processed[(arr_processed <= 1970) & (arr_processed > 1)] = 1
    arr_processed[(arr_processed >= 1970) & (arr_processed < 1980)] = 2
    arr_processed[(arr_processed >= 1980) & (arr_processed < 1990)] = 3
    arr_processed[(arr_processed >= 1990) & (arr_processed < 2000)] = 4
    arr_processed[(arr_processed >= 2000) & (arr_processed < 2010)] = 5
    arr_processed[(arr_processed >= 2010) & (arr_processed <= 2020)] = 6
    arr_processed[(arr_processed >= 2020)] = 0
    arr_processed[(arr_processed <0)] = 0
    return arr_processed
def get_ufz_type(arr_processed):
    arr_processed[(arr_processed >= 1) & (arr_processed <= 4)] = 0
    arr_processed[(arr_processed >= 5) & (arr_processed <= 9)] = 1
    return arr_processed

class Hongkong_dataset(torch.utils.data.Dataset):
    def __init__(self, mode,cache=False, augmentation=True):
        super(Hongkong_dataset, self).__init__()
        if mode=='test':
            MAIN_FOLDER = '../dataset/hk_building_age/val/'
        else:
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
        self.data_files = glob.glob(DATA_FOLDER)
        if mode=='test':
            self.data_files = random.sample(self.data_files, 100)
        # if mode == 'train':
        # # List of files
        #     #self.data_files = random.sample(glob.glob(DATA_FOLDER), 2075)
        #     self.data_files = glob.glob(DATA_FOLDER)
        #     with open("train.txt", "r", encoding="utf-8") as f:
        #         self.data_files +=[line.strip() for line in f if line.strip()]
        # else:
        #     with open("test.txt", "r", encoding="utf-8") as f:
        #         self.data_files = [line.strip() for line in f if line.strip()]

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
        height = height / 100.0  # normalize to 0-1
        height = height[np.newaxis, :, :]

        ufzs=[]
        for year in ['1990','2000','2010','2020']:
            if os.path.exists(self.UFZ_FOLDER+name+'ufz_'+year+'.tif'):
                ufz = io.imread(self.UFZ_FOLDER+name+'ufz_'+year+'.tif')
                ufz = np.asarray(ufz, dtype=np.float32)
                ufz=get_ufz_type(ufz)
                # unique_values1 = np.unique(ufz)
                # print(unique_values1)
                ufzs.append(ufz)
            else:
                ufzs.append(np.zeros((512, 512)))


        label = np.asarray(io.imread(self.LABEL_FOLDER+name+'class.tif'))
        label = label.astype(np.int64)
        
        label_id = get_year_type(label) #背景为0

        boundary_files = self.BOUNDARY_FOLDER+name+'mask.tif'
        boundary = np.asarray(io.imread(boundary_files))
        boundary = boundary.astype(np.int64)
        zero_mask = (label_id == 0)
        boundary[zero_mask] = 0
        boundary, instance_num = generate_instance_mask(boundary)


        boundary = boundary-1 #整张图的instance mask 从-1开始
        instances = extract_instance_masks(boundary) #转换为instance mask
        label_id[boundary == -1] = 0
        # instances_class=get_mask_classes(instances,label_id) #获得所有instance
        # random_int = random.randint(0, len(instances)-1)
        # one_instances =instances[random_int]
        # unique_values1 = np.unique(instance)
        # print(unique_values1)
        # print(instance_num)
        #convert_to_color(instance[0]-1, main_dir='.', name='instance_{}'.format(i))

        # Data augmentation
        ufzs = np.stack(ufzs, axis=0).astype(np.float32)
        if self.mode == 'train' and self.augmentation:
            data, boundary, height,label,label_id, ufzs = self.data_augmentation(data,boundary, height, label,label_id, ufzs)
        
        # ufzs = generate_first_impervious_year(ufzs).astype(np.float32)
        # zero_mask = np.repeat((instance == -1)[np.newaxis, :, :], repeats=3, axis=0)
        # save_img(data, './', name = "imgpre_{}".format(1))
        # data [zero_mask]= 0
        # save_img(data, './', name = "img_{}".format(1))
        # save_img(one_instances, './', name = "mask_{}".format(1))
        # convert_to_color(label-1, main_dir='.', name='instance_{}'.format(i))
        # if random.random() < 0.5:
        #     height[:]=0
        if self.mode == 'train' or self.mode == 'test':
            return (torch.from_numpy(data),
                    torch.from_numpy(data), #无用之前是instances表示每一个instance 的mask，但train里面没有用到，先放data占位
                    torch.from_numpy(height),
                    torch.from_numpy(ufzs),
                    torch.from_numpy(label_id)-1,
                    torch.from_numpy(boundary),
                    torch.from_numpy(label) #具体的年份，train的时候无用
                    )
        else:
            instances = np.array(instances)  
            return (torch.from_numpy(data),
                    torch.from_numpy(instances),
                    torch.from_numpy(height),
                    torch.from_numpy(ufzs),
                    torch.from_numpy(label_id)-1,
                    torch.from_numpy(boundary),
                    torch.from_numpy(label)
                    )

def extract_instance_masks(instance_id_tensor) -> dict:
    """
    从W×H的instance ID张量中，提取每个instance的二值mask
    :param instance_id_tensor: 形状(W, H)的tensor，像素值=instance编号（从0开始）
    :return: 字典，key=instance编号，value=对应二值mask（W×H的bool tensor，1=该instance区域）
    """
    # 1. 获取图中所有非重复的instance编号（排除全0背景，若0是背景则过滤，否则保留）
    unique_ids = np.unique(instance_id_tensor)
    # 备注：若0是背景（无意义instance），则过滤：
    unique_ids = unique_ids[unique_ids != -1]
    
    # 2. 向量化提取每个instance的二值mask（无循环）
    instance_masks = []
    for ins_id in unique_ids:
        # 生成该instance的二值mask：像素值==ins_id的位置为True
        mask = (instance_id_tensor == ins_id)
        instance_masks.append(mask)
    
    return instance_masks


def get_mask_classes(mask, label) -> np.ndarray:

    mask_classes = np.zeros(len(mask), dtype=label.dtype)
    for i in range(len(mask)):
        mask_i = mask[i]  # (W, H)
        label_masked = label[mask_i]  # (N,)，N为mask覆盖的像素数
        
        mask_classes[i] = mode(label_masked, keepdims=False).mode
    
    return mask_classes