import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter as P
import itertools
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
import os
from torch.nn.modules.loss import _Loss, _WeightedLoss


DATASET = 'hongkong' #amsterdam hongkong
MODEL = 'Dino' #Dino FTransUNet

# Parameters
## SwinFusion
# WINDOW_SIZE = (64, 64) # Patch size
WINDOW_SIZE = (512, 512) # Patch size

STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "/mnt/d/Jialu/dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10 # Number of samples in a mini-batch

#LABELS = ["<=1960", "1960<x<=1970", "1970<x<=1980", "1980<x<=1990", "1990<x<=2000", "2000<x<=2010", "2010<x<=2020"] # Label names
if DATASET=='amsterdam':
    LABELS = ["x<1980", "1980<=x<=2000", "2000<x"] # Label names
if DATASET=='hongkong':
    LABELS = ["<=1970", "1970<x<=1980", "1980<x<=1990","1990<x<=2000", "2000<x<=2010", "2000<x<=2020"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {-1 : (255, 255, 255), # Undefined (white)
           0 : (0, 0, 255),     # <=1960
           1 : (0, 255, 255),   # 1960<x<=1970
           2 : (0, 255, 0),     # 1970<x<=1980
           3 : (255, 255, 0),   # 1980<x<=1990
           4 : (255, 0, 0),     # 1990<x<=2000
           5 : (255, 0, 255),   # 2000<x<=2010
           6 : (0, 0, 0)}       # 2010<x<=2020 black

invert_palette = {v: k for k, v in palette.items()}

MODE = 'train'

LOSS = 'SEG'  #ORD
# LOSS = 'SEG+BDY'
# LOSS = 'SEG+OBJ'
# LOSS = 'SEG+BDY+OBJ'


def convert_to_color(arr_2d, main_dir,name, palette=palette):
    """ Numeric labels to RGB-color encoding """
    if isinstance(arr_2d, torch.Tensor):
        arr_2d = arr_2d.cpu().numpy()
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    color_img = Image.fromarray(arr_3d)  # 彩色图直接转换

    color_img.save(os.path.join(main_dir, name + ".jpg"))  # 保存彩色图

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def save_img(tensor, main_dir, name):
    name = os.path.join(main_dir, name + ".jpg")
    if tensor.shape[0]==1:
        tensor = tensor.repeat(3, 1, 1)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    if tensor.shape[0]==3:
        tensor = np.transpose(tensor, axes=(1, 2, 0))
    # im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (tensor * 255.).astype(np.uint8)
    im = Image.fromarray(im).save(name)


def object_process(object):
    ids = np.unique(object)
    new_id = 1
    for id in ids[1:]:
        object = np.where(object == id, new_id, object)
        new_id += 1
    return object

        
# We load one tile from the dataset and we display it
# img = io.imread('./ISPRS_dataset/Vaihingen/top/top_mosaic_09cm_area11.tif')
# fig = plt.figure()
# fig.add_subplot(121)
# plt.imshow(img)
#
# # We load the ground truth
# gt = io.imread('./ISPRS_dataset/Vaihingen/gts_for_participants/top_mosaic_09cm_area11.tif')
# fig.add_subplot(122)
# plt.imshow(gt)
# plt.show()
#
# # We also check that we can convert the ground truth into an array format
# array_gt = convert_from_color(gt)
# print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)


# Utils

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

class CrossEntropy2d_ignore(nn.Module):

    def __init__(self, size_average=True, ignore_label=-1):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight,reduction='mean')
        return loss
    

def manual_cross_entropy_with_soft_label(input: torch.Tensor, soft_label: torch.Tensor, dim: int = 1):
    """
    手动实现交叉熵损失（适配高斯模糊后的软标签）
    :param input: 模型输出（未经过softmax），形状(N,C)（分类）或(N,C,H,W)（分割）
    :param soft_label: 高斯模糊后的软标签，形状(N,C)（分类）或(N,C,H,W)（分割）
    :param dim: 类别维度（分类/分割均为1）
    :return: 标量损失值
    """
    # 1. 模型输出做log_softmax（数值稳定，避免log(0)）
    log_pred = F.log_softmax(input, dim=dim)
    
    # 2. 计算负对数似然：-∑(soft_label * log_pred) / 样本数
    # 逐元素相乘后求和，再除以总样本数（分类：N；分割：N*H*W）
    if len(input.shape) == 2:  # 分类任务：(N,C)
        num_samples = input.shape[0]
        loss = -torch.sum(soft_label * log_pred) / num_samples
    else:  # 分割任务：(N,C,H,W)
        num_samples = input.shape[0] * input.shape[1] * input.shape[2]
        loss = -torch.sum(soft_label * log_pred) / num_samples
    
    return loss

def pdf_fn(x):
  x_pdf = torch.exp( -(x)**2 /2  ) * 1/( torch.pi * torch.sqrt(torch.tensor(2)) )
  return x_pdf

def fast_label_to_dist(one_hot_label):
    """
    向量化生成label_dist，无任何for循环
    :param one_hot_label: 输入one-hot标签，形状(N, C)，N为样本数，C为类别数
    :return: label_dist，形状(N, C)，和原代码逻辑完全一致
    """
    # 步骤1：批量获取所有样本的目标索引t（替代torch.where+循环），形状(N,)
    target_idx = torch.argmax(one_hot_label, dim=1)  # one-hot找1的索引，比where快10倍+
    
    # 步骤2：生成位置索引矩阵（0到C-1），形状(1, C)，广播到(N, C)
    C = one_hot_label.shape[1]
    pos_idx = torch.arange(C, device=one_hot_label.device).unsqueeze(0)  # (1, C)
    
    # 步骤3：将target_idx广播到(N, C)，计算绝对距离（核心！等价于原代码的序列）
    target_idx_expand = target_idx.unsqueeze(1)  # (N, 1) → 广播到(N, C)
    label_dist = torch.abs(pos_idx - target_idx_expand)  # 绝对距离，形状(N, C)
    
    return label_dist

def loss_calc(pred, label,instanc_class, weights):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label= Variable(label.long()).cuda()
    instanc_class = Variable(instanc_class.long()).cuda()
    criterion_piexl = CrossEntropy2d_ignore().cuda()
    piexl_loss = criterion_piexl(pred[0],label,weights)
    instance_loss = CrossEntropy2d(pred[1],instanc_class)
    loss = piexl_loss
    return loss

    # n, c, h, w = pred.size()
    # target_mask = (label >= 0) * (label != -1)
    # label = label[target_mask]
    # pred = pred.transpose(1, 2).transpose(2, 3).contiguous()
    # pred = pred[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    # one_hot_label = F.one_hot(label, num_classes=N_CLASSES)
    # label = fast_label_to_dist(one_hot_label)
    # label = pdf_fn(label)
    # return manual_cross_entropy_with_soft_label(pred, label, dim=1)

def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight,reduction='mean')
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight,reduction='mean')
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def accuracy(input, target):
    valid_mask = target != -1   
    target = target[valid_mask]
    input = input[valid_mask]
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

class ObjectLoss(nn.Module):
  def __init__(self, max_object=50):
        super().__init__()
        self.max_object = max_object

  def forward(self, pred, gt):
    num_object = int(torch.max(gt)) + 1
    num_object = min(num_object, self.max_object)
    total_object_loss = 0

    for object_index in range(1,num_object):
        mask = torch.where(gt == object_index, 1, 0).unsqueeze(1).to('cuda')
        num_point = mask.sum(2).sum(2).unsqueeze(2).unsqueeze(2).to('cuda')
        avg_pool = mask / (num_point + 1)

        object_feature = pred.mul(avg_pool)

        avg_feature = object_feature.sum(2).sum(2).unsqueeze(2).unsqueeze(2).repeat(1,1,gt.shape[1],gt.shape[2])
        avg_feature = avg_feature.mul(mask)

        object_loss = torch.nn.functional.mse_loss(num_point * object_feature, avg_feature, reduction='mean')
        total_object_loss = total_object_loss + object_loss
      
    return total_object_loss
  
class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, _, _, _ = pred.shape
        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)
        class_map = pred.argmax(dim=1).cpu()  # Get Class Map with the Shape: [B, H, W]

        # boundary map
        gt_b = F.max_pool2d(
            1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt

        pred_b = F.max_pool2d(
            1 - class_map, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - class_map

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, 2, -1)
        pred_b = pred_b.view(n, 2, -1)
        gt_b_ext = gt_b_ext.view(n, 2, -1)
        pred_b_ext = pred_b_ext.view(n, 2, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


class WeightedOrdinalLoss(nn.Module):
    """
    加权有序损失（Ordinal Loss），适配建筑年龄有序分类任务
    支持：1. 经典Ordinal Loss核心逻辑 2. 线性/指数权重，强化近期建筑预测 3. 忽略背景像素
    """
    def __init__(self, num_classes=7, weight_type="linear", lambda_weight=0.1, background_label=-1):
        """
        初始化参数
        Args:
            num_classes: 建筑年龄类别总数（如8类：1950-1960至2020-2026）
            weight_type: 权重类型，"linear"（线性权重）或"exp"（指数权重）
            lambda_weight: 权重超参数，控制近期建筑权重强度
            background_label: 背景像素标签（不参与损失计算），默认0
        """
        super(WeightedOrdinalLoss, self).__init__()
        self.num_classes = num_classes  # 类别总数C
        self.weight_type = weight_type
        self.lambda_weight = lambda_weight
        self.background_label = background_label

    def _get_sample_weight(self, target):
        """
        计算每个像素的样本权重（近期建筑权重更高）
        Args:
            target: 真实标签张量，shape [B, H, W]（批次、高度、宽度）
        Returns:
            weight: 与target同shape的权重张量
        """
        # 1. 筛选建筑像素（排除背景），背景权重设为0
        valid_mask = (target != self.background_label).float()
        # 2. 提取建筑像素的类别序号（c_i：0~num_classes-1，近期建筑类别序号更大）
        class_indices = (target.float()) * valid_mask

        # 3. 计算权重（保证权重为正，且近期类别权重更高）
        if self.weight_type == "linear":
            # 线性权重：w_i = 1 + lambda_weight * c_i（近期类别序号大，权重线性增加）
            weight = 1.0 + self.lambda_weight * class_indices
        elif self.weight_type == "exp":
            # 指数权重：w_i = exp(lambda_weight * c_i)（近期类别权重指数级增加）
            weight = torch.exp(self.lambda_weight * class_indices)
        else:
            raise ValueError("weight_type仅支持 'linear' 或 'exp'")

        # 4. 背景像素权重置0（不参与损失计算）
        weight = weight * valid_mask
        return weight

    def _ordinal_label_conversion(self, target):
        """
        有序标签转换：将单标签 [B, H, W] 转为二元有序标签 [B, C-1, H, W]
        核心逻辑：若真实类别为c，那么对于k=0~C-2，k < c 时标签为1，k >= c 时标签为0
        Args:
            target: 真实标签张量，shape [B, H, W]
        Returns:
            ordinal_target: 二元有序标签，shape [B, num_classes-1, H, W]
        """
        B, H, W = target.shape
        C = self.num_classes

        # 1. 扩展维度，便于广播计算 [B, H, W] -> [B, 1, H, W]
        target_expanded = target.unsqueeze(1)
        # 2. 生成k值序列（0~C-2），shape [1, C-1, 1, 1]
        k_sequence = torch.arange(0, C-1, device=target.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # 3. 广播比较：k < c 时为1，否则为0（得到二元有序标签）
        ordinal_target = (k_sequence < target_expanded).float()

        # 4. 背景像素对应的有序标签置0（不参与损失计算）
        valid_mask = (target != self.background_label).unsqueeze(1).float()
        ordinal_target = ordinal_target * valid_mask

        return ordinal_target

    def forward(self, input, target):
        """
        前向传播：计算加权Ordinal Loss
        Args:
            input: 模型预测输出，shape [B, num_classes, H, W]（无需softmax，内部用sigmoid）
            target: 真实标签，shape [B, H, W]（每个像素值为0~num_classes-1，背景为background_label）
        Returns:
            loss: 加权有序损失值（标量）
        """
        B, C, H, W = input.shape
        assert C == self.num_classes, f"模型输出通道数{C}需与类别数{self.num_classes}一致"
        assert target.shape == (B, H, W), f"目标标签shape {target.shape}需与[B, H, W]一致"

        # 步骤1：标签转换：单标签 -> 二元有序标签 [B, C-1, H, W]
        ordinal_target = self._ordinal_label_conversion(target)

        # 步骤2：模型输出处理：取前C-1个通道（对应k=0~C-2的二元判断），并通过sigmoid激活
        # 原因：Ordinal Loss将多分类转为C-1个二元分类，每个分类判断"是否大于类别k"
        input_ordinal = input[:, :-1, :, :]  # 取前C-1个通道，shape [B, C-1, H, W]
        input_sigmoid = torch.sigmoid(input_ordinal)  # 转为0~1的概率值

        # 步骤3：计算每个像素的权重（近期建筑权重更高）
        sample_weight = self._get_sample_weight(target)  # [B, H, W]
        # 扩展权重维度，适配有序损失计算 [B, H, W] -> [B, 1, H, W]
        weight_expanded = sample_weight.unsqueeze(1)

        # 步骤4：计算二元交叉熵（BCE）损失（带权重）
        # BCE公式：-w * [y*log(p) + (1-y)*log(1-p)]
        bce_loss = F.binary_cross_entropy(input_sigmoid, ordinal_target, weight=weight_expanded, reduction='none')

        # 步骤5：平均化损失（仅对有效建筑像素计算）
        valid_pixel_num = torch.clamp(sample_weight.sum(), min=1.0)  # 避免除以0
        loss = (bce_loss.sum()) / valid_pixel_num

        return loss


def metrics(predictions, gts, label_values=LABELS):

    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" %(kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:5])
    print('mean MIoU: %.4f' % (MIoU))

    diff  = gts-predictions
    diff_squared = np.square(diff)
    mse = np.mean(diff_squared)
    rmse = np.sqrt(mse)
    print("RMSE : %.4f" % (rmse))
    print("---")

    return MIoU