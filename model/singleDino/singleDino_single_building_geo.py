import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import LULCEncoder,DSMEncoder
from torchvision.ops import DeformConv2d
from .dinov3 import LayerNorm2d,DINOv3,Decoder
#模态合并v1    
class TriModalAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.attention = nn.Sequential(
            nn.Conv2d(dim*3, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, 3, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, img, depth, lulc,modality_mask):
        weight = self.attention(torch.cat([img, depth, lulc], dim=1))
        img_w = weight[:,0:1,:,:] * img
        depth_w = weight[:,1:2,:,:] * depth
        lulc_w = weight[:,2:3,:,:] * lulc
        return self.norm(img_w + depth_w + lulc_w)
#模态合并v2  
class TriModalCrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 三个模态投影
        self.proj_img = nn.Conv2d(dim, dim, 1)
        self.proj_dsm = nn.Conv2d(dim, dim, 1)
        self.proj_lulc = nn.Conv2d(dim, dim, 1)

        # 🌟 LULC 置信门控（核心：自动学习是否信任LULC特征）
        self.lulc_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()  # 输出 0~1，自动抑制无效特征
        )

        # Cross-Attention 主体
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.norm = LayerNorm2d(dim)

    def forward(self, img, depth, lulc,modality_mask):
        # 步骤1：特征投影
        img = self.proj_img(img)
        dsm = self.proj_dsm(depth)
        lulc = self.proj_lulc(lulc)

        # 🌟 步骤2：LULC置信门控（核心！）
        # 模型自动学习：无效LULC → 权重趋近0
        lulc_weight = self.lulc_gate(lulc)
        lulc = lulc * lulc_weight

        # 步骤3：多模态上下文融合
        ctx = dsm + lulc

        # 步骤4：Cross-Attention
        B, C, H, W = img.shape
        q = self.q(img).flatten(2).transpose(1, 2)
        k = self.k(ctx).flatten(2).transpose(1, 2)
        v = self.v(ctx).flatten(2).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
        attn = attn.softmax(dim=-1)
        fused = torch.matmul(attn, v).transpose(1, 2).view(B, C, H, W)
        fused = self.proj_out(fused)

        # 残差 + 归一化
        out = self.norm(img + fused)
        return out

class DeformableConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.offset = nn.Conv2d(in_c, 2*9, 3, padding=1)
        self.deform = DeformConv2d(in_c, out_c, 3, padding=1)
        self.norm = LayerNorm2d(out_c)
    def forward(self, x):
        offset = self.offset(x)
        x = self.deform(x, offset)
        return self.norm(x)


class GeoConditionalAdaIN(nn.Module):
    """
    几何条件化自适应归一化（适配 1D 向量输入）
    输入: instance_feat [B, 64], geo_feat [B, 4]
    输出: fused_feat [B, 64]
    """
    def __init__(self, img_dim=64, geo_dim=4):
        super().__init__()
        self.norm = nn.LayerNorm(img_dim)
        self.img_dim = img_dim
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, geo_dim * 2),
            nn.LayerNorm(geo_dim * 2),
            nn.GELU(),
        )
        # 几何特征 → 自适应 γ, β
        self.geo_to_gamma = nn.Linear(geo_dim*2, img_dim)
        self.geo_to_beta  = nn.Linear(geo_dim*2, img_dim)

    def forward(self, instance_feat, geo_feat):
        B, C = instance_feat.shape  # [B, 64]
        feat_norm = self.norm(instance_feat)
        geo_feat = self.geo_proj(geo_feat)

        # 2. 几何条件生成 γ, β
        gamma = self.geo_to_gamma(geo_feat)  # [B,64]
        beta  = self.geo_to_beta(geo_feat)   # [B,64]

        # 3. AdaIN 核心：条件化调制
        fused_feat = feat_norm * gamma + beta  # [B,64]

        return fused_feat


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Conv2d(dim, 1, 1)
    def forward(self, x, mask):
        B, C, H, W = x.shape
        if H == 1 and W == 1:
            return x.squeeze()
        weight = self.attn(x).sigmoid() * mask
        weight = weight / (weight.sum((2,3), keepdim=True) + 1e-6)
        return (x * weight).sum((2,3))


class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()

        self.image_encoder = DINOv3(
            backbone=torch.hub.load(
                "/mnt/d/Jialu/buildingage/dinov3",
                'dinov3_vitl16',  # hubconf.py 里定义的函数
                # 'dinov3_vith16plus',  # 1280
                source='local',
                pretrained=False  # 我们用自定义 checkpoint
            ),
            interaction_indexes=[23]
        )

        encoder_channels = (256, 256, 256, 256)

        for n, value in self.image_encoder.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        self.neck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False, ),
            LayerNorm2d(512),
            DeformableConvBlock(512, 256),
            LayerNorm2d(256) )
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            LayerNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)
        self.deep_encoder =DSMEncoder()
        self.ufz_encoder =LULCEncoder()

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(decode_channels),
            nn.Dropout(0.15),
            nn.Linear(decode_channels, decode_channels),
            nn.GELU(),
            nn.BatchNorm1d(decode_channels),
            nn.Linear(decode_channels, num_classes)
        )
    
         
        self.fuse1 = TriModalAttentionFusion(256)
        self.fuse2 = TriModalAttentionFusion(256)
        self.fuse3 = TriModalAttentionFusion(256)
        self.fuse4 = TriModalAttentionFusion(256)
        self.AdaIN=GeoConditionalAdaIN()
        self.attentionpool=AttentionPool(64)

    def forward(self, x, depth, masks,ufzs,geo_feat):
        b, _, h, w = x.size()
        modality_mask = ufzs.flatten(1).all(dim=1, keepdim=True).float()
        depth_feats = self.deep_encoder(depth)
        ufzs_feats = self.ufz_encoder(ufzs)
        deepx = self.image_encoder(x)  # 256*1024  
        deepx = deepx[0].permute(0, 2, 1).view(b, 1024, 32, 32)
        ## 这个deepx可由interaction_indexes这个控制，配了一个UNetformer的解码器，自行修改
        deepx = self.neck(deepx)
        res1 = self.fpn1(deepx)   # 256 128 128 
        res2 = self.fpn2(deepx) # 256 64 64
        res3 = self.fpn3(deepx) # 256 32 32
        res4 = self.fpn4(deepx) # 256 16 16

        res1 = self.fuse1(res1, depth_feats[0], ufzs_feats[0],modality_mask)
        res2 = self.fuse2(res2, depth_feats[1], ufzs_feats[1],modality_mask)
        res3 = self.fuse3(res3, depth_feats[2], ufzs_feats[2],modality_mask)
        res4 = self.fuse4(res4, depth_feats[3], ufzs_feats[3],modality_mask)

        x_piexl,feat_map = self.decoder(res1, res2, res3, res4, h, w)
        # 遍历该图的所有mask

        B, d, H_feat, W_feat = feat_map.shape
        instance_feats=[]
        geos_list=[]
        for b in range(B):
            mask = masks[b, :, :] 
            feature = feat_map[b, :, :, :]
            building_ids = torch.unique(mask)
            building_ids = [id for id in building_ids if id != -1]
            for bid in building_ids:
                mask_bid = (mask == bid).float()  # 生成当前建筑的二值mask
                mask_bid = mask_bid.unsqueeze(0).unsqueeze(0)
                mask_interp = nn.functional.interpolate(
                    mask_bid, 
                    size=(H_feat,W_feat),  # 对齐特征图尺寸
                    mode='nearest',        # 双线性插值（适合mask）
                )
                mask_interp = mask_interp[0]

                pooled_feat = self.attentionpool(feature.unsqueeze(0), mask_interp.unsqueeze(0))
                instance_feats.append(pooled_feat)

                # 4. 收集几何特征
                geos_list.append(geo_feat[b,bid].unsqueeze(0))


                #feat_flat = feature.reshape(d, -1)  # (d, 128×128)
                #mask_flat = mask_interp.reshape(1, -1)  # (1, 128×128)
                # sum_m = torch.clamp(mask_flat.sum(), min=1e-6)
                # max_f = torch.max(feat_flat * mask_flat, dim=1)[0]
                # avg_f = torch.sum(feat_flat * mask_flat, dim=1) / sum_m
                #inst_feat = torch.cat([max_f, avg_f], dim=0)


        if len(instance_feats) == 0:
            return x_piexl, torch.zeros(0, 3, device=x.device)
        attention_f =torch.cat(instance_feats, dim=0)  # [N, C]
        geos=torch.cat(geos_list, dim=0)
        fused_feat = self.AdaIN(attention_f, geos) 
        inst_logits = self.classifier(fused_feat)  # (B×3)×num_classes
        logits_clamped = torch.clamp(inst_logits, min=-10.0, max=10.0)
        return x_piexl, logits_clamped 