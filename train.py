import numpy as np
import torch

torch.cuda.device_count()
import torch.optim as optim
from torch.autograd import Variable
from IPython.display import clear_output
from config import convert_to_color, mse_rmse, DATASET,save_img,metrics,metrics_sinple,WeightedOrdinalLoss,accuracy,N_CLASSES,WINDOW_SIZE,MODE,LOSS,WEIGHTS,DATASET,MODEL,loss_calc_only_instance,loss_calculate,NUM_INSTANCE
from dataset import get_dataloader
import os
from kmean import generate_image
print(MODEL + ', ' + MODE + ', ' + DATASET + ', ' + LOSS)
main_dir = './result/{}_{}'.format(MODEL, DATASET)

if not os.path.exists(main_dir):
    os.makedirs(main_dir)

if MODEL == 'Dino':
    from model.singleDino.singleDino_single_building import UNetFormer as singleDino
    net = singleDino(num_classes=N_CLASSES).cuda()
if MODEL == 'Dino_mask':
    from model.singleDino.singleDino_single_building_mask import UNetFormer as singleDino
    net = singleDino(num_classes=N_CLASSES).cuda()
if MODEL == 'Dino_height':
    from model.singleDino.singleDino_single_building_height import UNetFormer as singleDino
    net = singleDino(num_classes=N_CLASSES).cuda()
if MODEL == 'FTransUNet':
    from model.ftransunet.FUNet import VisionTransformer
    net = VisionTransformer(img_size=WINDOW_SIZE[0], num_classes=N_CLASSES).cuda()
if MODEL == 'Unetformer':
    from model.unetformer.unetformer import UNetFormer
    net = UNetFormer(num_classes=N_CLASSES).cuda()
if MODEL == 'STunet':
    from model.ST_Unet.vit_seg_modeling import ST_Unet
    net = ST_Unet(img_size=WINDOW_SIZE[0],num_classes=N_CLASSES).cuda()
if MODEL == 'AsymFormer':
    from model.asymformer.AsymFormer import B0_T
    net = B0_T(num_classes=N_CLASSES).cuda()
if MODEL == 'CMTFNet':
    from model.CMTFNet.CMTFNet import CMTFNet
    net = CMTFNet(num_classes=N_CLASSES).cuda()
if MODEL == 'ABCNet':
    from model.ABCNet.ABCNet import ABCNet
    net = ABCNet(num_classes=N_CLASSES).cuda()
if MODEL == 'CMX':
    from model.CMX.builder import EncoderDecoder
    net = EncoderDecoder(num_classes=N_CLASSES).cuda()
if MODEL == 'CMNeXt':
    from model.CMNeXt.cmnext import CMNeXt
    net = CMNeXt(num_classes=N_CLASSES).cuda()
if MODEL == 'MFNet':
    from model.MFNet.UNetFormer_MMSAM import UNetFormer
    net = UNetFormer(num_classes=N_CLASSES).cuda()
if MODEL == 'Segformer':
    from model.Segformer.segformer import SegFormer
    net = SegFormer(num_classes=N_CLASSES).cuda()
if MODEL == 'TransUNet':
    from model.TransUNet.vit_seg_modeling import VisionTransformer
    net = VisionTransformer(num_classes=N_CLASSES).cuda()

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)

# Load the datasets
train_set = get_dataloader(DATASET, 'train')
train_loader = torch.utils.data.DataLoader(train_set,batch_size=10,shuffle=True)

val_set = get_dataloader(DATASET, 'val')
val_loader = torch.utils.data.DataLoader(val_set,batch_size=1)

test_set = get_dataloader(DATASET, 'test')
test_loader = torch.utils.data.DataLoader(test_set,batch_size=10)
print("training : ", len(train_set))
print("val : ", len(val_set))
print("training : ", len(train_set))
print("test : ", len(test_set))

base_lr = 0.005
params_dict = dict(net.named_parameters())
params = []
print('lr: ', base_lr)
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 35], gamma=0.25)
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
    if output.dim()==1:
        output = output.unsqueeze(0)
    class_indices = torch.argmax(output, dim=1)
    return class_indices

def get_instance_metric(pred_instance, instance_label,instance_year, label):
    pred_instance= pred_instance.permute(0,2,3,1) # [1,H,W,C]
    all_build=[]
    correct_build=[]
    all_building_year=[]
    for j in range(pred_instance.shape[0]): #batch size
        for i in range(len(instance_label)): #每一个实例判断对不对
            instance_label_i = pred_instance[j][instance_label[i]]
            mean_tensor = instance_label_i.mean(dim=0)
            pred_instance_label = torch.argmax(mean_tensor)
            # pred_instance_label = torch.mode(instance_label_i)[0].item() # 该实例的预测类别
            label_i = torch.mode(label[j][instance_label[i]])[0].item() # 该实例的真实类别
            label_i_year = torch.mode(instance_year[j][instance_label[i]])[0].item() # 该实例的真实建造年份
            all_building_year.append(label_i_year)
            all_build.append(label_i)
            correct_build.append(pred_instance_label.cpu())
    return (all_build,correct_build, all_building_year)

def get_mask_number(pred_instance,masks):
    result = torch.zeros(masks.shape[1])
    for i in range(masks.shape[1]):
        instance = pred_instance[0][masks[0][i]]
        result[i] = torch.mode(instance)[0].item()
    return result

def test_loss(net, first=False,loader = val_loader): #计算test取最大值的loss有多少，大约为1.16，需要配合改一下get_instance_metric的correct值才能跑
    net.eval()
    total_loss=0.
    with torch.no_grad():
        for batch_idx, (data, mask, height,ufzs, target,boundary, label_year) in enumerate(loader):
            data, mask,height,ufzs, target,boundary, label_year = Variable(data.cuda()), Variable(mask.cuda()), Variable(height.cuda()),Variable(ufzs.cuda()), Variable(target.cuda()),Variable(boundary.cuda()), Variable(label_year.cuda())
            output = net(data, height, boundary, ufzs)
            instance_num,correct,all_building_year = get_instance_metric(output[0], mask[0],label_year, target)
            correct = torch.stack(correct, dim=0).cuda()
            new_output = []
            new_output.append(correct)
            new_output.append(correct)
            loss = loss_calc_only_instance(new_output, target,boundary)
            total_loss += loss.item()
        total_loss /= len(loader)
        print(test_loss)

def test_loss_train(net,loader = test_loader):
    net.eval()
    test_loss=0.
    for batch_idx, (data, mask, height,ufzs, target,boundary,label_year) in enumerate(loader):
        data, mask,height,ufzs, target,boundary = Variable(data.cuda()), Variable(mask.cuda()), Variable(height.cuda()),Variable(ufzs.cuda()), Variable(target.cuda()),Variable(boundary.cuda())
        with torch.no_grad():
            output = net(data, height, boundary, ufzs)
            loss_ce = loss_calc_only_instance(output, target,boundary, weights=None)
            test_loss += loss_ce.item()
    test_loss /= len(loader)
    print(test_loss)
    net.train()

def test(net, first=False,loader = val_loader,epoch=100):
    net.eval()
    all_preds = []
    all_gts = []
    all_build=[]
    all_build_year=[]
    correct_build=[]
    j=0
    mask_list = []
    feature_list = []
    labels = []
    with torch.no_grad():
        for batch_idx, (data, mask, height,ufzs, target,boundary, label_year) in enumerate(loader):
            data, mask,height,ufzs, target,boundary, label_year = Variable(data.cuda()), Variable(mask.cuda()), Variable(height.cuda()),Variable(ufzs.cuda()), Variable(target.cuda()),Variable(boundary.cuda()), Variable(label_year.cuda())
            output = net(data, height, boundary, ufzs)
            class_indices = get_result(output[0])
            # if batch_idx==0:
            #     for item in range(class_indices.shape[0]):
            #         class_indices[target == -1]=-1
            #         convert_to_color(class_indices[item], main_dir, name = "pred_{}".format(item))
            #         convert_to_color(target[item], main_dir, name = "gt_{}".format(item))
                    # save_img(data[item], main_dir, name = "img_{}".format(item))
                    # save_img(height[item], main_dir, name = "height_{}".format(item))
            instance_num,correct,all_building_year = get_instance_metric(output[0], mask[0],label_year, target)
            if torch.is_tensor(output[1]) and epoch>NUM_INSTANCE:
                correct = get_result(output[1]).cpu()

            # mask_list.append(boundary.cpu())
            # feature_list.append(output[1].cpu())
            # labels.append(target.cpu())

            valid_mask = target != -1
            target = target[valid_mask]
            class_indices = class_indices[valid_mask]
            if first and batch_idx>5:
                break
            
            # time_pred=[]
            # time_gt=[]
            # time_build=[]
            # time_correct_build=[]
            # time_build.append(instance_num)
            # time_correct_build.append(correct)
            # time_pred.append(class_indices.cpu().numpy())
            # time_gt.append(target.cpu().numpy())

            all_preds.append(class_indices.cpu().numpy())
            all_gts.append(target.cpu().numpy())
            all_build.append(instance_num)
            all_build_year.append(all_building_year)
            correct_build.append(correct)

            # accuracy = metrics_sinple(np.concatenate([p.ravel() for p in time_pred]),
            #                 np.concatenate([p.ravel() for p in time_gt]))
            # instance_accuracy = metrics_sinple(np.concatenate([p for p in time_correct_build]),
            #                 np.concatenate([p for p in time_build]))
            # if(instance_accuracy>25):
            #     all_preds.append(class_indices.cpu().numpy())
            #     all_gts.append(target.cpu().numpy())
            #     all_build.append(instance_num)
            #     correct_build.append(correct)
            #     file_list.append(file_name[0])
            # else:
            #     j+=1
            #     file_list2.append(file_name[0])

        # 一次性写入文件
        # content = "\n".join(file_list)
        # with open("output.txt", "w", encoding="utf-8") as f:
        #     f.write(content)
        # content2 = "\n".join(file_list2)
        # with open("output2.txt", "w", encoding="utf-8") as f:
        #     f.write(content2)

        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                            np.concatenate([p.ravel() for p in all_gts]))
        instance_accuracy = metrics(np.concatenate([p for p in correct_build]),
                            np.concatenate([p for p in all_build]))
        mse_rmse(np.concatenate([p for p in correct_build]),
                            np.concatenate([p for p in all_build_year]))
        #generate_image(mask_list, feature_list,labels)

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

def test_semantic(net,first=False, loader = val_loader,epoch=100):
    net.eval()
    all_preds = []
    all_gts = []
    # Switch the network to inference mode
    with torch.no_grad():
        for batch_idx, (data, mask, height,ufzs, target,boundary,label_year) in enumerate(loader):
            data, mask,height,ufzs, target = Variable(data.cuda()), Variable(mask.cuda()), Variable(height.cuda()),Variable(ufzs.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data, height, boundary, ufzs)
            class_indices = get_result(output[0])
            if batch_idx==0:
                for item in range(class_indices.shape[0]):
                    class_indices[target == -1]=-1
                    convert_to_color(class_indices[item], main_dir, name = "pred_{}".format(item))
                    convert_to_color(target[item], main_dir, name = "gt_{}".format(item))
                    # save_img(data[item], main_dir, name = "img_{}".format(item))
                    # save_img(height[item], main_dir, name = "height_{}".format(item))
            valid_mask = target != -1
            target = target[valid_mask]
            class_indices = class_indices[valid_mask]
            all_preds.append(class_indices.cpu().numpy())
            all_gts.append(target.cpu().numpy())
            if first and batch_idx>5:
                break
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                        np.concatenate([p.ravel() for p in all_gts]))

        return accuracy

def test_all(net, loader = val_loader):
    net.eval()
    all_building = np.zeros((341962, 6))
    # Switch the network to inference mode
    with torch.no_grad():
        for batch_idx, (data, mask, height,ufzs, id_mapping) in enumerate(loader):
            data, mask,height,ufzs = Variable(data.cuda()), Variable(mask.cuda()), Variable(height.cuda()),Variable(ufzs.cuda())
            optimizer.zero_grad()
            output = net(data, height, mask, ufzs)
            #class_indices = get_result(output[0])
            #instance_class=get_mask_number(class_indices,mask) #这是每个instance 的类别，没有6分类的值，目前这个代码有问题
            all_np_keys = [key for key in id_mapping.keys() if isinstance(key, np.int64)]
            for i in range(len(all_np_keys)):
                all_building[all_np_keys[i]]+=output[1][i].cpu().numpy()
            if batch_idx>10:
                break
            if batch_idx%100==0:
                print(batch_idx)
        max_ids = np.argmax(all_building, axis=1)
        all_zero_rows = np.all(all_building == 0, axis=1)
        max_ids[all_zero_rows] = -1
        np.savetxt(
            './all_building.txt',          # 保存路径
            max_ids,                       # 要保存的数组
            fmt="%d",           # 格式：整数（避免科学计数法）
            newline="\n"        # 每行一个值，换行符分隔
        )
    return max_ids


# 计算全模型梯度的总L2范数
def get_total_grad_norm(model):
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += torch.norm(param.grad) **2
    grad_norm = torch.sqrt(grad_norm)
    print(f"\n全模型梯度总范数：{grad_norm.item():.4f}")
    return grad_norm.item()

def train(net, optimizer, epochs,test_function,  scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    criterionor = WeightedOrdinalLoss(num_classes = N_CLASSES)
    for e in range(1, epochs + 1):
        # if e == 1:
        #     test_function(net, first=True)
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, mask, height,ufzs, target,boundary,label_year) in enumerate(train_loader):
            data,height,ufzs, target,boundary = Variable(data.cuda()), Variable(height.cuda()),Variable(ufzs.cuda()), Variable(target.cuda()),Variable(boundary.cuda())
            optimizer.zero_grad()
            output = net(data, height, boundary, ufzs)
            loss = loss_calculate(output, target,boundary,e)
            loss.backward()
            #total_grad_norm = get_total_grad_norm(net)
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
                    100. * batch_idx / len(train_loader), mean_losses[iter_], accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)
        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            MIoU = test_function(net,epoch=e)
            net.train()
            torch.save(net.state_dict(), main_dir + '/{}_epoch{}_{}.pth'.format(MODEL, e, MIoU))

test_function = test_semantic if DATASET=='amsterdam' else test

if MODE == 'train':
    # net.load_state_dict(torch.load('./Dino_epoch42_0.4564934817950233.pth'),strict=False) 
    train(net, optimizer, 70, test_function, scheduler)
if MODE == 'test':
    net.load_state_dict(torch.load('./Dino_epoch42_0.4564934817950233.pth'),strict=True) 
    net.eval()
    if DATASET=='global_hongkong':
        test_all(net,val_loader)
    else:
        MIoU = test_function(net, loader=val_loader)
        print("MIoU: ", MIoU)
