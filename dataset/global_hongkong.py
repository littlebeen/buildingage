import torch
import glob
import random
import os
import numpy as np
from skimage import io
from scipy.ndimage import label as label11
from config import convert_to_color,save_img
from scipy.stats import mode

def generate_instance_mask(label_2d):
    w, h = label_2d.shape
    
    # 步骤2：初始化实例mask（2D），背景初始为0
    instance_mask_2d = np.zeros_like(label_2d, dtype=np.int32)
    global_instance_id = 1  # 全局实例ID，从1开始编号
    
    # 步骤3：定义连通域规则（8连通：上下左右+四个对角线，可改为4连通[[0,1,0],[1,1,1],[0,1,0]]）
    connectivity = np.ones((3, 3), dtype=np.bool_)  # 8连通结构元
    
    # 步骤4：遍历每一类（0-5），逐类检测连通域（不连续区域）
    for cls in range(6):
        # 生成当前类的掩码：仅当前类像素为True，其余（-1/其他类）为False
        cls_mask = (label_2d == cls)
        if np.sum(cls_mask) < 200:  # 该类别无像素，跳过
            continue
        
        # 连通域检测：返回标记数组（同一连通域为相同编号）、连通域数量
        labeled_cls, cls_instance_num = label11(cls_mask, structure=connectivity)
        
        # 遍历当前类的每个实例（每个连通域），分配全局唯一ID
        for inst in range(1, cls_instance_num + 1):
            # 找到当前类的当前实例的像素位置，赋值为全局ID
            inst_pixel_pos = (labeled_cls == inst)
            instance_mask_2d[inst_pixel_pos] = global_instance_id
            global_instance_id += 1  # 全局ID自增，为下一个实例准备
    
    # 步骤5：恢复为1*w*h的3维结构，与输入维度一致
    instance_mask = instance_mask_2d[np.newaxis, :, :]  # 形状恢复为(1, w, h)
    total_instance_num = global_instance_id - 1  # 总实例数（最后一次自增后需减1）
    
    return instance_mask, total_instance_num


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
        instance, instance_num = generate_instance_mask(label-1)
        instance = instance[0]-1 #整张图的instance mask 从-1开始
        instances = extract_instance_masks(instance) #转换为instance mask
        instances_class=get_mask_classes(instances,label) #获得所有instance
        random_int = random.randint(0, len(instances)-1)
        one_instances =instances[random_int]
        # unique_values1 = np.unique(instance)
        # print(unique_values1)
        # print(instance_num)
        #convert_to_color(instance[0]-1, main_dir='.', name='instance_{}'.format(i))

        # Data augmentation
        if self.mode == 'train' and self.augmentation:
            data, one_instances, boundary, height, label,instance, ufzs[0],ufzs[1], ufzs[2], ufzs[3] = self.data_augmentation(data, one_instances,boundary, height, label,instance,ufzs[0],ufzs[1], ufzs[2], ufzs[3])
        height = height[np.newaxis, :, :]
        boundary = boundary[np.newaxis, :, :]
        ufzs = np.stack(ufzs, axis=0)
        #ufzs = generate_first_impervious_year(ufzs).astype(np.float32)
        label[instance == -1] = 0
        zero_mask = np.repeat((instance == -1)[np.newaxis, :, :], repeats=3, axis=0)
        data [zero_mask]= 0
        #save_img(data, './', name = "img_{}".format(1))
        if self.mode == 'train' :
              one_instances = one_instances[np.newaxis,:,:]
              return (torch.from_numpy(data),
                        torch.from_numpy(one_instances),
                        torch.from_numpy(height),
                        torch.from_numpy(ufzs-1),
                        torch.from_numpy(label)-1,
                        torch.from_numpy(np.array(instances_class[random_int]))-1)
        else:
            instances = np.array(instances)
            return (torch.from_numpy(data),
                    torch.from_numpy(instances),
                    torch.from_numpy(height),
                    torch.from_numpy(ufzs-1),
                    torch.from_numpy(label)-1,
                    torch.from_numpy(instances_class)-1)
    

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