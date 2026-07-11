import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import LULCEncoder,DSMEncoder
from torchvision.ops import DeformConv2d
from .dinov3 import LayerNorm2d,DINOv3,Decoder
from .singleDino_single_building_geo import GeoConditionalAdaIN,AttentionPool
from kmean import save_each_expert_heatmap_separately,save_feature_heatmaps_with_bar
#模态合并v1    
class SimpleTriModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        # 直接把3个模态concat，再用1x1卷积融合回dim维度
        self.fusion = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, img, depth, lulc, modality_mask=None):
        # 直接拼接三个模态
        fused = torch.cat([img, depth, lulc], dim=1)
        # 1x1卷积融合
        fused = self.fusion(fused)
        return self.norm(fused)  
    
#模态融合v4无cross attention    

class CosGuidedMoEFusion(nn.Module):
    def __init__(self, dim, num_experts=2, dropout=0.0):  # 专家减到2个！
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        # ----------------------
        # 极简 MoE 专家（超快）
        # ----------------------
        self.experts = nn.ModuleList([
            nn.Conv2d(dim*2, dim, kernel_size=1)  # 1x1卷积 = 最快
            for _ in range(num_experts)
        ])

        # ----------------------
        # 模态门控 + MoE门控（极简）
        # ----------------------
        self.modal_gate = nn.Conv2d(dim*3, num_experts, kernel_size=1)
        self.norm = LayerNorm2d(dim)

    def forward(self, img, dsm, lulc, modality_mask):
        B, C, H, W = img.shape
        
        # ======================
        # 1. 保留【模态掩码mask】创新点
        # ======================
        mask_spatial = modality_mask.unsqueeze(-1).unsqueeze(-1)  # [B,3,1,1]
        dsm = dsm * mask_spatial[:,1:2]  # 掩码控制dsm是否生效
        lulc = lulc * mask_spatial[:,2:3]# 掩码控制lulc是否生效

        # 辅助特征融合
        ctx = dsm + lulc

        # ======================
        # 2. 保留【轻量余弦引导】
        # ======================
        cos_sim = F.cosine_similarity(img.flatten(2), ctx.flatten(2), dim=1, eps=1e-8)
        #save_feature_heatmaps_with_bar(cos_sim.view(B,1,H,W), "cos_sim",color_bar=True)
        img_guided = img * (1 + 0.1 * cos_sim.view(B,1,H,W))  # 轻量不耗时
        #save_feature_heatmaps_with_bar(img_guided, "img_guide")
        # ======================
        # 3. 保留【MoE混合专家】创新点
        # ======================
        feat = torch.cat([img_guided, ctx], dim=1)
        
        # 轻量门控（无循环，无耗时操作）
        gate = self.modal_gate(torch.cat([img, dsm, lulc], dim=1))
        gate = torch.softmax(gate, dim=1)  # [B, num_experts, H, W]
        #save_each_expert_heatmap_separately(gate)

        # 专家融合（最快写法）
        fused = 0.0
        for i in range(self.num_experts):

            fused = fused + gate[:, i:i+1] * self.experts[i](feat)
            #t = self.experts[i](feat)
            #save_feature_heatmaps_with_bar(t, "expert_{}".format(i))
            #save_feature_heatmaps_with_bar(gate[:, i:i+1] * self.experts[i](feat), "gateexpert_{}".format(i))

        # 残差连接
        out = self.norm(img + fused)
        return out

class DSMlulcMoEFusion(nn.Module):
    def __init__(self, dim, num_experts=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Conv2d(dim * 2, dim, kernel_size=1)
            for _ in range(num_experts)
        ])

        self.modal_gate = nn.Conv2d(dim * 2, num_experts, kernel_size=1)
        self.norm = LayerNorm2d(dim)

    def forward(self, dsm, lulc, modality_mask):
        mask_spatial = modality_mask.unsqueeze(-1).unsqueeze(-1)  # [B,3,1,1]
        dsm = dsm * mask_spatial[:,1:2]
        lulc = lulc * mask_spatial[:,2:3]

        feat = torch.cat([dsm, lulc], dim=1)
        gate = self.modal_gate(feat)
        gate = torch.softmax(gate, dim=1)

        fused = 0.0
        for i in range(self.num_experts):
            fused = fused + gate[:, i:i+1] * self.experts[i](feat)

        out = self.norm(fused)
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


class GeoSimpleConcat(nn.Module):
    """
    改进版直接拼接：
    1. 先对齐维度
    2. 特征归一化
    3. 两层MLP（防止梯度消失）
    能收敛，比原版强很多
    """
    def __init__(self, img_dim=64, geo_dim=4, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or img_dim * 2
        
        # 先把地理/属性特征升维，和图像特征维度匹配（关键改进）
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, img_dim),
            nn.LayerNorm(img_dim),
            nn.GELU()
        )
        
        # 拼接后的融合MLP（两层，比单层强太多）
        self.fusion = nn.Sequential(
            nn.Linear(img_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, img_dim)
        )

    def forward(self, instance_feat, geo_feat):
        # instance_feat: 图像/实例特征 [B, img_dim]
        # geo_feat:     几何/属性特征  [B, geo_dim]
        
        # 统一维度 + 归一化
        geo_proj = self.geo_proj(geo_feat)  # [B, img_dim]
        
        # 拼接
        fused = torch.cat([instance_feat, geo_proj], dim=-1)
        
        # 融合映射
        out = self.fusion(fused)
        return out

class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 encoder_channels=256,
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
        encoder_channels_list = (encoder_channels, encoder_channels, encoder_channels, encoder_channels)

        for n, value in self.image_encoder.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
            else:
                value.requires_grad = True

        self.neck = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False, ),
            LayerNorm2d(512),
            DeformableConvBlock(512, encoder_channels),
            LayerNorm2d(encoder_channels) )
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels, encoder_channels, kernel_size=2, stride=2),
            LayerNorm2d(encoder_channels),
            nn.GELU(),
            nn.ConvTranspose2d(encoder_channels, encoder_channels, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels, encoder_channels, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder = Decoder(encoder_channels_list, decode_channels, dropout, window_size, num_classes)
        self.deep_encoder =DSMEncoder(encoder_channels)
        self.ufz_encoder =LULCEncoder(encoder_channels)

    
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(decode_channels),
            nn.Dropout(0.15),
            nn.Linear(decode_channels, decode_channels),
            nn.GELU(),
            nn.BatchNorm1d(decode_channels),
            nn.Linear(decode_channels, num_classes)
        )
        

         
        self.fuse1 = CosGuidedMoEFusion(encoder_channels)
        self.fuse2 = CosGuidedMoEFusion(encoder_channels)
        self.fuse3 = CosGuidedMoEFusion(encoder_channels)
        self.fuse4 = CosGuidedMoEFusion(encoder_channels)
        #消融实验：直接拼接+MLP融合
        #self.AdaIN=GeoSimpleConcat(img_dim=decode_channels)
        self.AdaIN=GeoConditionalAdaIN(img_dim=decode_channels)
        self.attentionpool=AttentionPool(decode_channels)

    def forward(self, x, depth, masks,ufzs,geo_feat=None):
        b, _, h, w = x.size()
        modality_mask = ufzs.flatten(1).all(dim=1, keepdim=True).float()
        ones_b1 = torch.ones(b, 2, device=x.device)
        zeros_b2 = torch.zeros(b, 2, device=x.device)
        #dsm_tensor = torch.bernoulli(torch.full((b, 1), 0.7, device=x.device))
        modality_mask = torch.cat([ones_b1, modality_mask], dim=1)

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

        B, d, W_feat, H_feat = feat_map.shape
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

                # feat_flat = feature.reshape(d, -1)  # (d, 128×128)
                # mask_flat = mask_interp.reshape(1, -1)  # (1, 128×128)
                # sum_m = torch.clamp(mask_flat.sum(), min=1e-6)
                # avg_f = torch.sum(feat_flat * mask_flat, dim=1) / sum_m
                
                # instance_feats.append(avg_f.unsqueeze(0))


                # zeros_b2 = torch.zeros(4, device=x.device)
                # geos_list.append(zeros_b2.unsqueeze(0))

                geos_list.append(geo_feat[b,bid].unsqueeze(0))

        attention_f =torch.cat(instance_feats, dim=0)  # [N, C]
        geos=torch.cat(geos_list, dim=0)
        fused_feat = self.AdaIN(attention_f, geos) 
        inst_logits = self.classifier(fused_feat)  # (B×3)×num_classes
        logits_clamped = torch.clamp(inst_logits, min=-10.0, max=10.0)
        return x_piexl, logits_clamped 
    

