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
import matplotlib.pyplot as plt

DATASET = 'amsterdam' #amsterdam hongkong global_hongkong
MODEL = 'Dino_moe' #Dino Dino_improve Dino_moe Dino_geo Dino_geo Unetformer AsymFormer CMTFNet ABCNet CMX CMNeXt Segformer TransUNet CMT FTransDeepLab Unet
#FTransUNet STunet MFNet太慢了
MODE = 'train'
PRETRAIN =''
LOSS = 'SEG'  #ORD SEG
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
if DATASET=='hongkong' or DATASET=='global_hongkong':
    LABELS = [ "x<=1970","1970<x<=1980", "1980<x<=1990","1990<x<=2000", "2000<x<=2010", "2000<x<=2020"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
WEIGHTS[0] = 0.3
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


def hotmap(feature):

    # 取每个通道的均值，变成一张热力图 (128,128)
    heatmap = feature.mean(dim=0).detach().cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='jet')
    plt.colorbar(shrink=0.8)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('hotmap.png', dpi=300, bbox_inches='tight')

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


def focalLoss(inputs, targets, gamma=2,num_classes=N_CLASSES):
    # inputs: 模型输出 [N, num_classes]
    # targets: 标签 [N]
    
    log_prob = F.log_softmax(inputs, dim=-1)
    prob = torch.exp(log_prob)
    
    # 核心：难样本加权
    focal_weight = (1 - prob) ** gamma
    log_prob = focal_weight * log_prob
    
    # 转成 one-hot
    targets = F.one_hot(targets, num_classes=num_classes).float()
    
    # 计算损失
    loss = - (targets * log_prob).sum(dim=-1)
    return loss.mean()

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
        loss = F.cross_entropy(predict, target, weight=None,reduction='mean')
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


def get_instance_label(labels,boundarys):
    B,w,h = labels.shape
    building_label = []
    for b in range(B):
        boundary = boundarys[b, :, :] 
        building_ids = torch.unique(boundary)
        building_ids = [id for id in building_ids if id != -1]
        for bid in building_ids:
            mask_bid = torch.where(boundary == bid)
            label_i = torch.mode(labels[b][mask_bid])[0].item() 
            building_label.append(label_i)
    combined_tensor = torch.tensor(building_label).cuda()
    return combined_tensor

def loss_calculate(output,target,boundary,epoch):
    if 'Dino' in MODEL:
        if torch.is_tensor(output[1]):
            loss_ce = loss_calc_instance(output, target,boundary, WEIGHTS)
        else:
            loss_ce = loss_calc(output, target,boundary, WEIGHTS)
    else:
        loss_ce = loss_calc(output, target,boundary, WEIGHTS)
    return loss_ce

def loss_calc(pred, label,boundary, weights):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label= Variable(label.long()).cuda()
    criterion_piexl = CrossEntropy2d_ignore().cuda()
    piexl_loss = criterion_piexl(pred[0],label,weights)
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


def loss_calc_instance(pred, label,boundary, weights):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label= Variable(label.long()).cuda()
    criterion_piexl = CrossEntropy2d_ignore().cuda()
    piexl_loss = criterion_piexl(pred[0],label,weights)
    instanc_class = get_instance_label(label,boundary)
    if LOSS=='ORD':
        instance_loss = ordinalageloss(pred[1],instanc_class)
    elif LOSS =='SEG':
        instance_loss = focalLoss(pred[1],instanc_class)
    else:
        assert False
    loss = instance_loss+piexl_loss
    return loss

def loss_calc_only_instance(pred, label,boundary, weights=None):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label= Variable(label.long()).cuda()
    instanc_class = get_instance_label(label,boundary)
    instance_loss = CrossEntropy2d(pred[1],instanc_class)
    loss = instance_loss
    return loss

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

    return MIoU


def ordinalageloss(logits, targets):
    loss_ce = F.cross_entropy(logits, targets)
    num_bins = logits.shape[1]
    ordinal_targets = (torch.arange(num_bins, device=logits.device)[None, :] < targets[:, None]).float()
    loss_ord = F.binary_cross_entropy_with_logits(logits, ordinal_targets)
    return 0.2 * loss_ce + 0.8 * loss_ord



def metrics_sinple(predictions, gts, label_values=LABELS):

    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    # print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    return accuracy


def change_back_year(predictions):
    # 将预测的类别标签转换回对应的建筑年代
    year_mapping = {
        0: 1965,
        1: 1975,
        2: 1988,
        3: 1995,
        4: 2005,
        5: 2015,
    }
    # 使用向量化操作进行转换
    vectorized_mapping = np.vectorize(year_mapping.get)
    predicted_years = vectorized_mapping(predictions)
    return predicted_years

def mse_rmse(predictions, gts):
    zero_mask = (gts != 0)
    gts = gts[zero_mask]
    predictions = predictions[zero_mask] 
    predictions = change_back_year(predictions)
    diff  = gts-predictions
    diff_squared = np.square(diff)
    mse = np.mean(diff_squared)
    rmse = np.sqrt(mse)
    print("RMSE : %.4f" % (rmse))

    diff  = np.abs(gts-predictions)
    mae = np.mean(diff)
    print("MAE : %.4f" % (mae))
    print("---")