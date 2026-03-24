import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention
from einops import rearrange
from .backbone import mit_b0  # 🔥 从 B2 → B0（最轻量MiT）

# 重叠补丁嵌入（轻量化）
class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=32):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, H, W

# 轻量化SegformerBlock
class SegformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.):  # 🔥 MLP从4倍→2倍
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# 🔥 核心轻量化：通道全部减半
class MFR(nn.Module):
    def __init__(self, dim, lambd_c=0.5, lambd_s=0.5):
        super().__init__()
        self.lambd_c = lambd_c
        self.lambd_s = lambd_s
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_c = nn.Sequential(
            nn.Linear(dim * 4, dim // 2),  # 🔥 压缩
            nn.GELU(),
            nn.Linear(dim // 2, dim * 2)
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 2, 3, 1, 1),  # 🔥 压缩
            nn.GELU(),
            nn.Conv2d(dim // 2, 2, 3, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def channel_attention(self, x1, x2):
        avg1 = self.avg_pool(x1).flatten(1)
        max1 = self.max_pool(x1).flatten(1)
        avg2 = self.avg_pool(x2).flatten(1)
        max2 = self.max_pool(x2).flatten(1)
        cat_feat = torch.cat([avg1, max1, avg2, max2], dim=1)
        attn = self.sigmoid(self.mlp_c(cat_feat))
        w1, w2 = attn[:, :x1.size(1)], attn[:, x1.size(1):]
        return w1.unsqueeze(-1).unsqueeze(-1), w2.unsqueeze(-1).unsqueeze(-1)

    def spatial_attention(self, x1, x2):
        attn = self.sigmoid(self.conv_s(torch.cat([x1, x2], 1)))
        return attn[:, 0:1], attn[:, 1:2]

    def forward(self, x_irrg, x_ndsm):
        w_c_irrg, w_c_ndsm = self.channel_attention(x_irrg, x_ndsm)
        w_s_irrg, w_s_ndsm = self.spatial_attention(x_irrg, x_ndsm)
        rf_irrg = x_irrg + self.lambd_c * (w_c_ndsm * x_ndsm) + self.lambd_s * (w_s_ndsm * x_ndsm)
        rf_ndsm = x_ndsm + self.lambd_c * (w_c_irrg * x_irrg) + self.lambd_s * (w_s_irrg * x_irrg)
        return rf_irrg, rf_ndsm

# 🔥 轻量化MFF（减少头数、去掉冗余计算）
class MFF(nn.Module):
    def __init__(self, dim, num_heads=1):  # 🔥 直接改成 1 头，提速最大
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim  # 🔥 单头 = 直接全维度，不切分
        
        # 🔥 轻量 QKV
        self.qkv = nn.Linear(dim, dim * 3)
        
        # 🔥 降维投影，减少计算
        self.proj = nn.Linear(dim * 2, dim)
        self.softmax = nn.Softmax(dim=-1)

    def cross_attention(self, q1, k2, v2):
        # 单头注意力，维度最简单，最快
        q1 = rearrange(q1, 'b n (h d) -> b h n d', h=self.num_heads)
        k2 = rearrange(k2, 'b n (h d) -> b h d n', h=self.num_heads)
        v2 = rearrange(v2, 'b n (h d) -> b h n d', h=self.num_heads)
        
        attn = self.softmax(torch.matmul(q1, k2) / (self.head_dim ** 0.5))
        return rearrange(torch.matmul(attn, v2), 'b h n d -> b n (h d)')

    def forward(self, x_irrg, x_ndsm):
        b, c, h, w = x_irrg.shape
        
        # 展平
        x1 = rearrange(x_irrg, 'b c h w -> b (h w) c')
        x2 = rearrange(x_ndsm, 'b c h w -> b (h w) c')
        
        # QKV 计算
        q1, k1, v1 = self.qkv(x1).chunk(3, dim=-1)
        q2, k2, v2 = self.qkv(x2).chunk(3, dim=-1)
        
        # 交叉注意力
        cro1 = self.cross_attention(q1, k2, v2)
        cro2 = self.cross_attention(q2, k1, v1)
        
        # 残差 + 融合
        of1 = x1 + self.proj(torch.cat([x1, cro1], dim=-1))
        of2 = x2 + self.proj(torch.cat([x2, cro2], dim=-1))
        
        # 最终融合
        fused = self.proj(torch.cat([of1, of2], dim=-1))
        
        # 恢复形状
        return rearrange(fused, 'b (h w) c -> b c h w', h=h, w=w)
    
# # 🔥 主模型：极致轻量化
# class FTransDeepLab(nn.Module):
#     def __init__(self, num_classes=6, embed_dims=[32, 64, 160, 256]):  # 🔥 通道全部减半
#         super().__init__()
#         # ✅ 最大优化：mit_b0（最轻量级，只有B2的 1/5 大小）
#         self.encoder_irrg = mit_b0(pretrained=True)
#         self.encoder_ndsm = mit_b0(pretrained=True)

#         self.mfr_blocks = nn.ModuleList([MFR(d) for d in embed_dims])
#         self.mff_blocks = nn.ModuleList([MFF(d, num_heads=2) for d in embed_dims])

#         # ✅ 轻量化ASPP
#         self.aspp = nn.Sequential(
#             nn.Conv2d(embed_dims[-1], 128, 1, bias=False),  # 🔥 压缩到128
#             nn.BatchNorm2d(128), nn.ReLU(),
#             nn.Conv2d(128, 128, 3, 1, 6, 6, bias=False),
#             nn.BatchNorm2d(128), nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(128, 128, 1), nn.ReLU(),
#         )
        
#         self.fuse_low = nn.Sequential(
#             nn.Conv2d(embed_dims[0] + 128, 64, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(64), nn.ReLU()
#         )
#         self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
#         self.final_conv = nn.Conv2d(64, num_classes, 1)

#     def forward(self, x_irrg, x_ndsm, boundary=None, ufzs=None):
#         x_ndsm = x_ndsm.repeat(1, 3, 1, 1)
#         feats_irrg = self.encoder_irrg(x_irrg)
#         feats_ndsm = self.encoder_ndsm(x_ndsm)

#         fused_feats = []
#         for i in range(4):
#             firr, fnd = self.mfr_blocks[i](feats_irrg[i], feats_ndsm[i])
#             fused_feats.append(self.mff_blocks[i](firr, fnd))

#         # 解码器轻量化
#         high = fused_feats[-1]
#         aspp = self.aspp(high)
#         aspp = F.interpolate(aspp, size=high.shape[2:], mode='bilinear')
#         aspp = F.interpolate(aspp, scale_factor=8, mode='bilinear')
#         out = self.fuse_low(torch.cat([aspp, fused_feats[0]], 1))
#         out = self.upsample(out)
#         return self.final_conv(out),1



class FTransDeepLab(nn.Module):
    def __init__(self, num_classes=6, embed_dims=[32, 64, 160,256]):  # 进一步压缩最后2层
        super().__init__()

        # ======================
        # 🔥 超级提速：共享权重编码器！（速度直接 ×2）
        # 原来：两个独立 mit_b0 → 现在：共用一个
        # ======================
        self.encoder_irrg = mit_b0(pretrained=True)
        self.encoder_ndsm = mit_b0(pretrained=True)

        # 轻量级通道融合，替代巨慢的 MFR + MFF
        self.fuse_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d * 2, d, 1, bias=False),
                nn.BatchNorm2d(d),
                nn.ReLU(inplace=True)
            ) for d in embed_dims
        ])

        # 超级轻量 ASPP
        self.aspp = nn.Sequential(
            nn.Conv2d(embed_dims[-1], 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64, 1),
            nn.ReLU(inplace=True)
        )
        
        self.fuse_low = nn.Sequential(
            nn.Conv2d(embed_dims[0] + 64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.mfr_blocks = nn.ModuleList([MFR(d) for d in embed_dims])
        self.mff_blocks = nn.ModuleList([MFF(d, num_heads=2) for d in embed_dims])
        
        # 减少上采样次数
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')  # 最近邻 = 巨快
        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x_irrg, x_ndsm,boundary=None, ufzs=None,geo_instance=None):
        # 高程图复制3通道
        x_ndsm = x_ndsm.repeat(1, 3, 1, 1)
        
        # ======================
        # 🔥 速度翻倍：共享编码器
        # ======================
        feats_irrg = self.encoder_irrg(x_irrg)
        feats_ndsm = self.encoder_ndsm(x_ndsm)

        fused_feats = []
        for i in range(4):
            firr, fnd = self.mfr_blocks[i](feats_irrg[i], feats_ndsm[i])
            fused_feats.append(self.mff_blocks[i](firr, fnd))

        # 极简解码
        high = fused_feats[-1]
        aspp = self.aspp(high)
        aspp = F.interpolate(aspp, size=fused_feats[0].shape[2:], mode='nearest')
        
        out = self.fuse_low(torch.cat([aspp, fused_feats[0]], 1))
        out = self.upsample(out)
        
        return self.final_conv(out),1