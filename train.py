import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random, time
import itertools
import matplotlib.pyplot as plt
import torch
torch.cuda.device_count()
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from IPython.display import clear_output
from model.singleDino import UNetFormer as singleDino
from config import *
from dataset import get_dataloader


DATASET = 'hongkong' #amsterdam hongkong
MODEL = 'Dino'
print(MODEL + ', ' + MODE + ', ' + DATASET + ', ' + LOSS)
main_dir = './result/{}_{}'.format(MODEL, DATASET)

if not os.path.exists(main_dir):
    # os.makedirs()：创建文件夹，支持创建多级嵌套目录（如 "a/b/c/d"）
    os.makedirs(main_dir)

if MODEL == 'Dino':
    net = singleDino(num_classes=N_CLASSES).cuda()

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)

# Load the datasets
train_set = get_dataloader(DATASET, 'train')
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)


val_set = get_dataloader(DATASET, 'val')
val_loader = torch.utils.data.DataLoader(val_set,batch_size=BATCH_SIZE)
print("training : ", len(train_set))
print("val : ", len(val_set))

base_lr = 0.01
LBABDA_BDY = 0.1
LBABDA_OBJ = 1.0
print("LBABDA_BDY: ", LBABDA_BDY)
print("LBABDA_OBJ: ", LBABDA_OBJ)
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)

def get_result(output,threshold=0.5):
    input_ordinal = output[:, :-1, :, :]  # 取前C-1个通道，shape [1, C-1, H, W]
    sigmoid_probs = torch.sigmoid(input_ordinal)  # 激活为0~1的概率值
    
    # 3. 二值化：基于阈值得到二元判断结果（0/1）
    binary_predictions = (sigmoid_probs >= threshold).float()  # shape [1, C-1, H, W]
    
    # 4. 逐像素求和，得到建筑年龄类别序号（核心步骤）
    #    dim=1：对C-1个二元判断结果求和，shape [1, H, W] -> [H, W]
    class_indices = torch.sum(binary_predictions, dim=1).squeeze(0).long()  # 移除批次维度
    return class_indices

def test(net):
   
    all_preds = []
    all_gts = []
    # Switch the network to inference mode
    with torch.no_grad():
        for batch_idx, (data, boundary, object, target) in enumerate(val_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            outs = output.data.cpu().numpy()
            class_indices = get_result(output)
            if batch_idx==0:
                for item in range(class_indices.shape[0]):
                    class_indices[target == -1]=-1
                    convert_to_color(class_indices[item], main_dir, name = "pred_{}".format(item))
                    convert_to_color(target[item], main_dir, name = "gt_{}".format(item))
            valid_mask = target != -1   
            target = target[valid_mask]
            class_indices = class_indices[valid_mask]
            all_preds.append(class_indices.cpu().numpy())
            all_gts.append(target.cpu().numpy())
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                        np.concatenate([p.ravel() for p in all_gts]))
    return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=3):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.30
    criterionb = BoundaryLoss()
    criteriono = ObjectLoss()
    criterionor = WeightedOrdinalLoss(num_classes = N_CLASSES)
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, boundary, object, target) in enumerate(train_loader):
            continue
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            # loss_ce = loss_calc(output, target, weights)
            # loss_boundary = criterionb(output, boundary)
            # loss_object = criteriono(output, object)
            loss_ordinal = criterionor(output, target)
   
            # if LOSS == 'SEG':
            #     loss = loss_ce
            # elif LOSS == 'SEG+BDY':
            #     loss = loss_ce + loss_boundary * LBABDA_BDY
            # elif LOSS == 'SEG+OBJ':
            #     loss = loss_ce + loss_object * LBABDA_OBJ
            # elif LOSS == 'SEG+BDY+OBJ':
            #     loss = loss_ce + loss_boundary * LBABDA_BDY + loss_object * LBABDA_OBJ
            if LOSS == 'ORD':
                loss = loss_ordinal
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])



            if iter_ % 50 == 0:
                clear_output()
                pred = get_result(output)
                pred = pred.data.cpu().numpy()[0]
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            net.eval()
            MIoU = test(net)
            net.train()
            if MIoU > MIoU_best:
                torch.save(net.state_dict(), main_dir + '/{}_epoch{}_{}'.format(MODEL, e, MIoU))
                MIoU_best = MIoU

if MODE == 'train':
    train(net, optimizer, 50, scheduler)
elif MODE == 'test':
    if DATASET == 'Vaihingen':
        net.load_state_dict(torch.load(main_dir + '/YOUR_MODEL')) # sam
        net.eval()
        MIoU, all_preds, all_gts = test(net)
        print("MIoU: ", MIoU)