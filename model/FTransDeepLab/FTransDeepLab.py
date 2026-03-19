import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention
from einops import rearrange, repeat
from timm.models.segformer import Segformer, MiT_B2

# 重叠补丁嵌入（Overlap Patch Embedding）- 论文Stacked Segformer核心组件
class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=1024, patch_size=7, stride=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W)
        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, H, W

# 单个Segformer Block（含Transformer+LN+MLP）
class SegformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class StackedSegformerEncoder(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], depths=[3, 3, 3, 3]):
        super().__init__()
        self.stages = nn.ModuleList()
        # 4个阶段的Overlap Patch Embedding参数（论文3.2节）
        patch_params = [
            (7, 4, 3),  # 阶段1: K=7, S=4, P=3
            (3, 2, 1),  # 阶段2: K=3, S=2, P=1
            (3, 2, 1),  # 阶段3: K=3, S=2, P=1
            (3, 2, 1)   # 阶段4: K=3, S=2, P=1
        ]
        for i in range(4):
            # 重叠补丁嵌入
            patch_embed = OverlapPatchEmbed(
                patch_size=patch_params[i][0],
                stride=patch_params[i][1],
                padding=patch_params[i][2],
                in_chans=in_chans if i==0 else embed_dims[i-1],
                embed_dim=embed_dims[i]
            )
            # Segformer Block堆叠
            blocks = nn.Sequential(*[SegformerBlock(embed_dims[i], num_heads[i]) for _ in range(depths[i])])
            self.stages.append(nn.ModuleDict({'patch_embed': patch_embed, 'blocks': blocks}))
        # 加载MiT-B2预训练权重（论文实验设置）
        self._init_pretrained(MiT_B2(pretrained=True))

    def _init_pretrained(self, pretrained_model):
        # 对齐预训练权重与自定义模块
        for i, stage in enumerate(self.stages):
            stage['patch_embed'].proj.load_state_dict(pretrained_model.stages[i].patch_embed.proj.state_dict())
            stage['patch_embed'].norm.load_state_dict(pretrained_model.stages[i].patch_embed.norm.state_dict())
            stage['blocks'].load_state_dict(pretrained_model.stages[i].blocks.state_dict())

    def forward(self, x):
        feats = []  # 保存4个阶段的特征
        Hs, Ws = [], []
        for stage in self.stages:
            x, H, W = stage['patch_embed'](x)
            x = stage['blocks'](x)
            feats.append(x)
            Hs.append(H)
            Ws.append(W)
            # 恢复为2D特征图，方便后续MFR/MFF处理
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return feats, Hs, Ws
    
class MFR(nn.Module):
    def __init__(self, dim, lambd_c=0.5, lambd_s=0.5):
        super().__init__()
        self.lambd_c = lambd_c
        self.lambd_s = lambd_s
        # 通道注意力模块（共享MLP）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp_c = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim * 2)
        )
        # 空间注意力模块（共享卷积）
        self.conv_s = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 2, kernel_size=3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def channel_attention(self, x1, x2):
        # x1: IRRG特征, x2: nDSM特征 (B, C, H, W)
        b, c, h, w = x1.shape
        # 池化聚合空间信息
        avg1, max1 = self.avg_pool(x1).flatten(1), self.max_pool(x1).flatten(1)
        avg2, max2 = self.avg_pool(x2).flatten(1), self.max_pool(x2).flatten(1)
        # 拼接双模态+双池化特征
        cat_feat = torch.cat([avg1, max1, avg2, max2], dim=1)  # (B, 4C)
        # 共享MLP生成注意力权重
        attn = self.mlp_c(cat_feat)  # (B, 2C)
        attn = self.sigmoid(attn)
        w1, w2 = attn[:, :c], attn[:, c:]  # 分别对应IRRG/nDSM通道权重
        w1, w2 = w1.unsqueeze(-1).unsqueeze(-1), w2.unsqueeze(-1).unsqueeze(-1)
        return w1, w2

    def spatial_attention(self, x1, x2):
        # 拼接双模态特征
        cat_feat = torch.cat([x1, x2], dim=1)  # (B, 2C, H, W)
        # 共享卷积生成空间注意力权重
        attn = self.conv_s(cat_feat)  # (B, 2, H, W)
        attn = self.sigmoid(attn)
        w1, w2 = attn[:, 0:1, :, :], attn[:, 1:2, :, :]  # 分别对应IRRG/nDSM空间权重
        return w1, w2

    def forward(self, x_irrg, x_ndsm):
        # 计算通道和空间注意力权重
        w_c_irrg, w_c_ndsm = self.channel_attention(x_irrg, x_ndsm)
        w_s_irrg, w_s_ndsm = self.spatial_attention(x_irrg, x_ndsm)
        # 论文公式1/2：双模态双向特征校正
        rf_irrg = x_irrg + self.lambd_c * (w_c_ndsm * x_ndsm) + self.lambd_s * (w_s_ndsm * x_ndsm)
        rf_ndsm = x_ndsm + self.lambd_c * (w_c_irrg * x_irrg) + self.lambd_s * (w_s_irrg * x_irrg)
        return rf_irrg, rf_ndsm
    

class MFF(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # QKV线性映射（双模态共享）
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # 交叉注意力投影层
        self.proj = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def cross_attention(self, q1, k2, v2):
        # q1: 模态1查询, k2/v2: 模态2键/值 (B, N, C)
        q1 = rearrange(q1, 'b n (h d) -> b h n d', h=self.num_heads)
        k2 = rearrange(k2, 'b n (h d) -> b h d n', h=self.num_heads)
        v2 = rearrange(v2, 'b n (h d) -> b h n d', h=self.num_heads)
        # 注意力分数计算
        attn = torch.matmul(q1, k2) / (self.head_dim ** 0.5)
        attn = self.softmax(attn)
        # 加权融合值特征
        out = torch.matmul(attn, v2)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

    def forward(self, x_irrg, x_ndsm):
        b, c, h, w = x_irrg.shape
        # 展平为序列特征 (B, H*W, C)
        x1 = rearrange(x_irrg, 'b c h w -> b (h w) c')
        x2 = rearrange(x_ndsm, 'b c h w -> b (h w) c')
        # QKV映射
        q1, k1, v1 = self.qkv(x1).chunk(3, dim=-1)
        q2, k2, v2 = self.qkv(x2).chunk(3, dim=-1)
        # 论文公式7/8：交叉注意力计算
        cro1 = self.cross_attention(q1, k2, v2)  # IRRG融合nDSM信息
        cro2 = self.cross_attention(q2, k1, v1)  # nDSM融合IRRG信息
        # 论文公式9/10：特征拼接+投影
        of1 = x1 + self.proj(torch.cat([x1, cro1], dim=-1))
        of2 = x2 + self.proj(torch.cat([x2, cro2], dim=-1))
        # 论文公式11：拼接融合+恢复2D特征
        fused = torch.cat([of1, of2], dim=-1)
        fused = self.proj(fused)  # 降维至原通道
        fused = rearrange(fused, 'b (h w) c -> b c h w', h=h, w=w)
        return fused
    
class FTransDeepLab(nn.Module):
    def __init__(self, num_classes=6, in_chans=3, embed_dims=[64, 128, 320, 512]):
        super().__init__()
        self.num_classes = num_classes
        # 双模态Segformer编码器（IRRG和nDSM各一个）
        self.encoder_irrg = StackedSegformerEncoder(in_chans=in_chans, embed_dims=embed_dims)
        self.encoder_ndsm = StackedSegformerEncoder(in_chans=in_chans, embed_dims=embed_dims)
        # 4个阶段的MFR+MFF模块（对应编码器4个阶段输出）
        self.mfr_blocks = nn.ModuleList([MFR(dim) for dim in embed_dims])
        self.mff_blocks = nn.ModuleList([MFF(dim) for dim in embed_dims])
        # DeepLabv3+解码器（ASPP+上采样+低维特征融合）
        self.aspp = nn.Sequential(
            nn.Conv2d(embed_dims[-1], embed_dims[-1], 1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[-1], embed_dims[-1], 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(embed_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[-1], embed_dims[-1], 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(embed_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[-1], embed_dims[-1], 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(embed_dims[-1]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dims[-1], embed_dims[-1], 1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        )
        # 低维特征融合+上采样
        self.fuse_low = nn.Sequential(
            nn.Conv2d(embed_dims[0] + embed_dims[-1], embed_dims[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(embed_dims[0], num_classes, 1)

    def forward(self, x_irrg, x_ndsm):
        # 双模态编码器前向
        feats_irrg, Hs, Ws = self.encoder_irrg(x_irrg)
        feats_ndsm, _, _ = self.encoder_ndsm(x_ndsm)
        fused_feats = []
        # 逐阶段MFR+MFF特征校正与融合
        for i in range(4):
            # 恢复编码器输出为2D特征图
            feat_irrg = rearrange(feats_irrg[i], 'b (h w) c -> b c h w', h=Hs[i], w=Ws[i])
            feat_ndsm = rearrange(feats_ndsm[i], 'b (h w) c -> b c h w', h=Hs[i], w=Ws[i])
            # MFR校正
            rf_irrg, rf_ndsm = self.mfr_blocks[i](feat_irrg, feat_ndsm)
            # MFF融合
            fused_feat = self.mff_blocks[i](rf_irrg, rf_ndsm)
            fused_feats.append(fused_feat)
        # DeepLabv3+解码：ASPP处理最高维特征
        x = self.aspp(fused_feats[-1])
        # 融合最低维特征（论文解码器策略）
        low_feat = F.interpolate(fused_feats[0], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_feat], dim=1)
        x = self.fuse_low(x)
        # 上采样恢复输入尺寸+分类
        x = self.upsample(x)
        x = self.final_conv(x)
        return x