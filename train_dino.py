from model.singleDino import UNetFormer as singleDino

net = singleDino(num_classes=N_CLASSES).cuda()
