import torch
import glob
import random
import os
import numpy as np
from skimage import io
from scipy.ndimage import label as label11
from config import convert_to_color,save_img
from scipy.stats import mode
from .hongkong import get_ufz_type

def generate_instance_mask(mask, ignore_bg = 0,min_pixel=20):

    non_bg_mask = (mask != ignore_bg)
    non_bg_ids = mask[non_bg_mask]
    unique_ids, counts = np.unique(non_bg_ids, return_counts=True)
    id_counts = dict(zip(unique_ids, counts))

    invalid_ids = [id for id in unique_ids if id_counts[id] < min_pixel]
    for invalid_id in invalid_ids:
        mask[mask==invalid_id]=0


    unique_ids = np.unique(mask[mask != ignore_bg])
    unique_ids = np.sort(unique_ids)  # 排序确保映射顺序稳定
    
    # 2. 建立新旧编号的映射表（原大数值→连续0~n）
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    # 背景值保留原值（映射表中补充）
    id_mapping[ignore_bg] = -1


    
    # 3. 向量化替换：用映射表将原mask值转为新编号（核心！无需循环）
    # 步骤1：创建掩码，区分背景和instance
    bg_mask = (mask == ignore_bg)
    # 步骤2：对非背景区域做映射（利用np.vectorize向量化）
    vectorized_map = np.vectorize(lambda x: id_mapping[x])
    remapped_mask = vectorized_map(mask)
    
    # 验证：确保背景值未被修改（可选，增强鲁棒性）
    assert np.all(remapped_mask[bg_mask] == -1), "背景值映射错误"
    
    return remapped_mask, id_mapping


class Hongkong_dataset(torch.utils.data.Dataset):
    def __init__(self, mode,cache=False, augmentation=True):
        super(Hongkong_dataset, self).__init__()
        if mode =='train':
            self.data_files=[1]
            return
        MAIN_FOLDER = '../dataset/global_Hongkong/'
        DATA_FOLDER = MAIN_FOLDER + 'image/tdop*.tif'
        self.BOUNDARY_FOLDER = MAIN_FOLDER + 'mask/'
        self.HEIGHT_FOLDER = MAIN_FOLDER + 'height/'
        self.UFZ_FOLDER = MAIN_FOLDER + 'ufz/'
        self.mode = mode
        self.augmentation = augmentation
        self.cache = cache
        self.max_num=0

        # List of files
        self.data_files = glob.glob(DATA_FOLDER)

    def __len__(self):
        # Default epoch size is 10 000 samples
        return len(self.data_files)

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

        boundary_files = self.BOUNDARY_FOLDER+name+'mask.tif'
        boundary = np.asarray(io.imread(boundary_files))
        boundary = boundary.astype(np.int64)

        boundary, id_mapping = generate_instance_mask(boundary)
        instances = extract_instance_masks(boundary) #转换为instance mask
        # unique_values1 = np.unique(instance)
        # print(unique_values1)
        # print(instance_num)
        #convert_to_color(instance[0]-1, main_dir='.', name='instance_{}'.format(i))

        ufzs = np.stack(ufzs, axis=0).astype(np.float32)
        #save_img(data, './', name = "img_{}".format(1))
        instances = np.array(instances)
        return (torch.from_numpy(data),
                torch.from_numpy(instances),
                torch.from_numpy(height),
                torch.from_numpy(ufzs),
                torch.from_numpy(boundary),
                id_mapping)
    

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