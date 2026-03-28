def get_model(N_CLASSES,WINDOW_SIZE,MODEL):
    if MODEL == 'Dino_final':
        from model.singleDino.singleDino_single_building_final import UNetFormer as singleDino
        net = singleDino(num_classes=N_CLASSES).cuda()
    if MODEL == 'Dino':
        from model.singleDino.singleDino_single_building import UNetFormer as singleDino
        net = singleDino(num_classes=N_CLASSES).cuda()
    if MODEL == 'Dinoa':
        from model.singleDino.singleDino_single_building_improvea import UNetFormer as singleDino
        net = singleDino(num_classes=N_CLASSES).cuda()
    if MODEL == 'Dino_mask':
        from model.singleDino.singleDino_single_building_mask import UNetFormer as singleDino
        net = singleDino(num_classes=N_CLASSES).cuda()
    if MODEL == 'Dino_geo':
        from model.singleDino.singleDino_single_building_geo import UNetFormer as singleDino
        net = singleDino(num_classes=N_CLASSES).cuda()
    if MODEL == 'Dino_improve':
        from model.singleDino.singleDino_single_building_improve import UNetFormer as singleDino
        net = singleDino(num_classes=N_CLASSES).cuda()
    if MODEL == 'Dino_moe':
        from model.singleDino.singleDino_single_building_moe import UNetFormer as singleDino
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
    if MODEL == 'CMT':
        from model.CMT.cmt import CMT
        net = CMT(num_classes=N_CLASSES).cuda()
    if MODEL == 'A2FPN':
        from model.A2FPN.a2fpn import A2FPN
        net = A2FPN(num_classes=N_CLASSES).cuda()
    if MODEL == 'Unet':
        from model.Unet.Unet import Unet
        net = Unet(num_classes=N_CLASSES).cuda()
    if MODEL == 'Deeplab':
        from model.Deeplab.Deeplab import DeepLabV3
        net = DeepLabV3(num_classes=N_CLASSES).cuda()
    if MODEL == 'FTransDeepLab':
        from model.FTransDeepLab.FTransDeepLab import FTransDeepLab
        net = FTransDeepLab(num_classes=N_CLASSES).cuda()
    return net