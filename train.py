import numpy as np
import torch

torch.cuda.device_count()
import torch.optim as optim
from torch.autograd import Variable
from IPython.display import clear_output
from config import convert_to_color, save_img,metrics,WeightedOrdinalLoss,accuracy,loss_calc,N_CLASSES,WINDOW_SIZE,BATCH_SIZE,MODE,LOSS,WEIGHTS,DATASET,MODEL
from dataset import get_dataloader
import os

print(MODEL + ', ' + MODE + ', ' + DATASET + ', ' + LOSS)
main_dir = './result/{}_{}mask'.format(MODEL, DATASET)

if not os.path.exists(main_dir):
    # os.makedirs()：创建文件夹，支持创建多级嵌套目录（如 "a/b/c/d"）
    os.makedirs(main_dir)

if MODEL == 'Dino':
    from model.singleDino.singleDino_single import UNetFormer as singleDino
    net = singleDino(num_classes=N_CLASSES).cuda()
if MODEL == 'FTransUNet':
    from model.ftransunet.FUNet import VisionTransformer
    net = VisionTransformer(img_size=WINDOW_SIZE[0], num_classes=N_CLASSES).cuda()

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)

# Load the datasets
train_set = get_dataloader(DATASET, 'train')
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

test_set = get_dataloader(DATASET, 'test')
test_loader = torch.utils.data.DataLoader(test_set,batch_size=1)

val_set = get_dataloader(DATASET, 'val')
val_loader = torch.utils.data.DataLoader(val_set,batch_size=1)
print("training : ", len(train_set))
print("val : ", len(val_set))
print("test : ", len(test_set))

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

# def get_result(output,threshold=0.5):
#     input_ordinal = output[:, :-1, :, :]  # 取前C-1个通道，shape [1, C-1, H, W]
#     sigmoid_probs = torch.sigmoid(input_ordinal)  # 激活为0~1的概率值
    
#     # 3. 二值化：基于阈值得到二元判断结果（0/1）
#     binary_predictions = (sigmoid_probs >= threshold).float()  # shape [1, C-1, H, W]
    
#     # 4. 逐像素求和，得到建筑年龄类别序号（核心步骤）
#     #    dim=1：对C-1个二元判断结果求和，shape [1, H, W] -> [H, W]
#     class_indices = torch.sum(binary_predictions, dim=1).squeeze(0).long()  # 移除批次维度
#     return class_indices

def get_result(output,threshold=0.5):
    class_indices = torch.argmax(output, dim=1)
    return class_indices

# def get_instance_metric(pred_instance, instance_label, label):

#     all_build=[]
#     correct_build=[]
#     for j in range(pred_instance.shape[0]): #batch size
#         instance_num = len(torch.unique(instance_label[j]))-1 # 减去背景类0，背景为-1
#         for i in range(instance_num): #每一个实例判断对不对
#             instance_mask = (instance_label[j] == i)
#             instance_label_i = pred_instance[j][instance_mask]
#             pred_instance_label = torch.mode(instance_label_i)[0].item() # 该实例的预测类别
#             label_i = torch.mode(label[j][instance_mask])[0].item() # 该实例的真实类别
#             all_build.append(label_i)
#             correct_build.append(pred_instance_label)
#             # if pred_instance_label == torch.mode(label_i)[0].item():
#             #     correct_build+=1  
    
#     return (all_build,correct_build)

def test(net, loader = val_loader):
   
    all_preds = []
    all_gts = []
    all_build=[]
    correct_build=[]
    # Switch the network to inference mode
    with torch.no_grad():
        for batch_idx, (data, mask, height,ufzs, target,instanc_class) in enumerate(loader):
            data, mask,height,ufzs, target = Variable(data.cuda()), Variable(mask.cuda()), Variable(height.cuda()),Variable(ufzs.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data, height, mask, ufzs)
            class_indices = get_result(output[0])
            instance_indices = get_result(output[1])
            if batch_idx==0:
                for item in range(class_indices.shape[0]):
                    class_indices[target == -1]=-1
                    convert_to_color(class_indices[item], main_dir, name = "pred_{}".format(item))
                    convert_to_color(target[item], main_dir, name = "gt_{}".format(item))
                    # save_img(data[item], main_dir, name = "img_{}".format(item))
                    # save_img(height[item], main_dir, name = "height_{}".format(item))
            # instance_num,correct = get_instance_metric(class_indices, mask, target)
            # all_build.append(instance_num)
            # correct_build.append(correct)
            all_build.append(instanc_class.cpu().numpy())
            correct_build.append(instance_indices.cpu().numpy())
            valid_mask = target != -1   
            target = target[valid_mask]
            class_indices = class_indices[valid_mask]
            all_preds.append(class_indices.cpu().numpy())
            all_gts.append(target.cpu().numpy())
            break
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                        np.concatenate([p.ravel() for p in all_gts]))
        inatance_accuracy = metrics(np.concatenate([p.ravel() for p in correct_build]),
                        np.concatenate([p.ravel() for p in all_build]))
        # unique_vals, val_counts = np.unique(np.concatenate([p.ravel() for p in all_gts]), return_counts=True)
        # print("="*50)
        # print(f"mask数组共包含 {len(unique_vals)} 种唯一值")
        # print("值 | 对应像素数量")
        # print("-"*20)
        # for val, count in zip(unique_vals, val_counts):
        #     print(f"{val:>d} | {count:>8d} 个像素")
        # print("="*50)
        # unique_vals, val_counts = np.unique(np.concatenate(all_build), return_counts=True)
        # print("="*50)
        # print(f"mask数组共包含 {len(unique_vals)} 种唯一值")
        # print("值 | 对应像素数量")
        # print("-"*20)
        # for val, count in zip(unique_vals, val_counts):
        #     print(f"{val:>d} | {count:>8d} 个像素")
        # print("="*50)
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=3):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.30
    criterionor = WeightedOrdinalLoss(num_classes = N_CLASSES)
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, mask, height,ufzs, target,instanc_class) in enumerate(train_loader):
            data, mask,height,ufzs, target = Variable(data.cuda()), Variable(mask.cuda()), Variable(height.cuda()),Variable(ufzs.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data, height, mask, ufzs)
            if LOSS == 'SEG':
                loss_ce = loss_calc(output, target,instanc_class, weights)
                loss = loss_ce
            if LOSS == 'ORD':
                loss_ordinal = criterionor(output, target)
                loss = loss_ordinal
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 50 == 0:
                clear_output()
                pred = get_result(output[0])
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
                torch.save(net.state_dict(), main_dir + '/{}_epoch{}_{}.pth'.format(MODEL, e, MIoU))
                MIoU_best = MIoU

if MODE == 'train':
    train(net, optimizer, 50, scheduler)
if MODE == 'test':
    #net.load_state_dict(torch.load('/mnt/d/Jialu/buildingage/result/Dino_hongkongheight/Dino_epoch48_0.31642549688233534.pth')) # sam
    net.eval()
    MIoU = test(net,loader= val_loader)
    print("MIoU: ", MIoU)
