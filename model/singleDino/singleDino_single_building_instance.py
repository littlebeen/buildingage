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
    def forward(self, img, depth, lulc):
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

    def forward(self, img, depth, lulc):
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
#模态合并v3     
class CosGuidedMoECrossAttentionFusion_Robust_Lite(nn.Module):
    def __init__(self, dim, num_experts=2, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        # 输入投影
        self.proj_img = nn.Conv2d(dim, dim, 1)
        self.proj_dsm = nn.Conv2d(dim, dim, 1)
        self.proj_lulc = nn.Conv2d(dim, dim, 1)

        # 模态门控
        self.gate_dsm = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.gate_lulc = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

        # ======================
        # 🔥 单头 Cross Attention 专家（无多头！无num_heads！）
        # ======================
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                "q_proj": nn.Conv2d(dim, dim, 1),   # Q: img
                "kv_proj": nn.Conv2d(dim, dim * 2, 1), # K,V: ctx
                "out_proj": nn.Conv2d(dim, dim, 1),
            }) for _ in range(num_experts)
        ])

        # MoE门控
        self.gate_net = nn.Sequential(
            nn.Conv2d(dim * 3 + 3, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, num_experts, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Softmax(dim=1)
        )

        self.norm = LayerNorm2d(dim)

    def forward(self, img, dsm, lulc, modality_mask):
        B, C, H, W = img.shape
        device = img.device

        # 模态掩码
        ones_b1 = torch.ones(B, 1, device=device)
        dsm_tensor = torch.bernoulli(torch.full((B, 1), 0.7, device=device))
        modality_mask = torch.cat([ones_b1, dsm_tensor, modality_mask], dim=1)
        mask_spatial = modality_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        mask_dsm = mask_spatial[:, 1:2]
        mask_lulc = mask_spatial[:, 2:3]

        # 投影 + 掩码
        img = self.proj_img(img)
        dsm = self.proj_dsm(dsm) * mask_dsm
        lulc = self.proj_lulc(lulc) * mask_lulc
        dsm = dsm * self.gate_dsm(dsm)
        lulc = lulc * self.gate_lulc(lulc)
        ctx = dsm + lulc

        # 余弦引导
        has_aux = (modality_mask[:, 1] + modality_mask[:, 2]).clamp(0, 1).view(B,1,1,1)
        cos_sim = F.cosine_similarity(img.flatten(2), ctx.flatten(2), dim=1).view(B,1,H,W) * has_aux
        cos_guide = cos_sim.flatten(2)  # [B, 1, HW]

        # MoE门控
        gate_in = torch.cat([img, dsm, lulc, mask_spatial.float()], dim=1)
        g = self.gate_net(gate_in)

        # ======================
        # 🔥 核心：纯单头 Cross Attention（无多头！）
        # ======================
        def attn(q, k, v, guide):
            # 输入: [B,C,H,W]
            q = q.flatten(2)          # [B, C, HW]
            k = k.flatten(2)          # [B, C, HW]
            v = v.flatten(2)          # [B, C, HW]

            # 单头交叉注意力
            attn_score = torch.matmul(q.transpose(1,2), k) * (self.dim ** -0.5)
            attn_score = attn_score + guide
            attn_score = attn_score.softmax(dim=-1)

            out = torch.matmul(v, attn_score.transpose(1,2))  # [B, C, HW]
            return out.view(B, C, H, W)  # ✅ 正确形状

        # 专家融合
        fused = 0
        for i, exp in enumerate(self.experts):
            q = exp["q_proj"](img)
            k, v = exp["kv_proj"](ctx).chunk(2, 1)
            o = attn(q, k, v, cos_guide)
            o = exp["out_proj"](o)
            fused = fused + g[:,i:i+1] * o

        out = self.norm(img + fused)
        return out
#模态融合v4无cross attention    
class CosGuidedMoEFusion(nn.Module):
    def __init__(self, dim, num_experts=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        # 输入投影
        self.proj_img = nn.Conv2d(dim, dim, 1)
        self.proj_dsm = nn.Conv2d(dim, dim, 1)
        self.proj_lulc = nn.Conv2d(dim, dim, 1)

        # 模态门控
        self.gate_dsm = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.gate_lulc = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

        # ======================
        # 🔥 单头 Cross Attention 专家（无多头！无num_heads！）
        # ======================
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, padding=1),
            ) for _ in range(num_experts)
        ])

        # MoE门控
        self.gate_net = nn.Sequential(
            nn.Conv2d(dim * 3 + 3, dim // 2, 1),
            nn.GELU(),
            nn.Conv2d(dim // 2, num_experts, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Softmax(dim=1)
        )

        self.norm = LayerNorm2d(dim)

    def forward(self, img, dsm, lulc, modality_mask):
        B, C, H, W = img.shape
        device = img.device

        # 模态掩码
        ones_b1 = torch.ones(B, 1, device=device)
        dsm_tensor = torch.bernoulli(torch.full((B, 1), 0.7, device=device))
        modality_mask = torch.cat([ones_b1, dsm_tensor, modality_mask], dim=1)
        mask_spatial = modality_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
        mask_dsm = mask_spatial[:, 1:2]
        mask_lulc = mask_spatial[:, 2:3]

        # 投影 + 掩码
        img = self.proj_img(img)
        dsm = self.proj_dsm(dsm) * mask_dsm
        lulc = self.proj_lulc(lulc) * mask_lulc
        dsm = dsm * self.gate_dsm(dsm)
        lulc = lulc * self.gate_lulc(lulc)
        ctx = dsm + lulc

        # 余弦引导
        has_aux = (modality_mask[:, 1] + modality_mask[:, 2]).clamp(0, 1).view(B,1,1,1)
        cos_sim = F.cosine_similarity(img.flatten(2), ctx.flatten(2), dim=1).view(B,1,H,W) * has_aux
        img_guided = img * (1 + cos_sim)  # 余弦引导直接乘进去，简单高效

        # MoE门控
        gate_in = torch.cat([img, dsm, lulc, mask_spatial.float()], dim=1)
        g = self.gate_net(gate_in)


        # 专家融合
        fused = 0.0
        feat = torch.cat([img_guided, ctx], dim=1)  # 引导特征 + 上下文
        for i, exp in enumerate(self.experts):
            out = exp(feat)
            fused = fused + g[:, i:i+1] * out

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
            nn.BatchNorm1d(2 * decode_channels),
            nn.Dropout(0.15),
            nn.Linear(2 * decode_channels, decode_channels),
            nn.GELU(),
            nn.BatchNorm1d(decode_channels),
            nn.Linear(decode_channels, num_classes)
        )
        

         
        self.fuse1 = CosGuidedMoEFusion(256)
        self.fuse2 = CosGuidedMoEFusion(256)
        self.fuse3 = CosGuidedMoEFusion(256)
        self.fuse4 = CosGuidedMoEFusion(256)

    def forward(self, x, depth, masks,ufzs):
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

        B, d, W_feat, H_feat = feat_map.shape
        buildings=[]
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
                    size=(W_feat, H_feat),  # 对齐特征图尺寸
                    mode='bilinear',        # 双线性插值（适合mask）
                    align_corners=True     # 避免边缘失真，推荐设置
                )
                mask_interp = mask_interp.squeeze()
                
                feat_flat = feature.reshape(d, -1)  # (d, 128×128)
                #feat_flat = feature.flatten(1)
                mask_flat = mask_interp.reshape(1, -1)  # (1, 128×128)
                sum_m = torch.clamp(mask_flat.sum(), min=1e-6)
                max_f = torch.max(feat_flat * mask_flat, dim=1)[0]
                avg_f = torch.sum(feat_flat * mask_flat, dim=1) / sum_m

                inst_feat = torch.cat([max_f, avg_f], dim=0)
                inst_feat = F.normalize(inst_feat, p=2, dim=0)

                buildings.append(inst_feat)

        inst_logits = self.classifier(torch.stack(buildings, dim=0))  # (B×3)×num_classes
        logits_clamped = torch.clamp(inst_logits, min=-10.0, max=10.0)
        return x_piexl, logits_clamped 
    

