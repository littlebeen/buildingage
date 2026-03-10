from torch.nn import functional as F
from .mix_transformer import OverlapPatchEmbed, mit_b0
from .convnext import asym_convnext_tiny as convnext_tiny
from .MLPDecoder import DecoderHead

import os
import torch
from torch import nn


def channel_shuffle(x, groups: int):
    batchsize, N, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, N, groups, channels_per_group)

    # Transpose operation is not valid for 5D tensor, so we need to use permute
    x = x.permute(0, 1, 3, 2).contiguous()

    # flatten
    x = x.view(batchsize, N, -1)

    return x


class Cross_Atten_Lite_split(nn.Module):
    def __init__(self, inc1, inc2):
        super(Cross_Atten_Lite_split, self).__init__()
        self.midc1 = inc1 // 4
        self.midc2 = inc2 // 4
        self.need_pad = self.midc1 < self.midc2
        self.pad_size = self.midc2 - self.midc1 if self.need_pad else 0

        self.bn_x1 = nn.BatchNorm2d(inc1)
        self.bn_x2 = nn.BatchNorm2d(inc2)

        self.kq1 = nn.Linear(inc1, self.midc2 * 2)
        self.kq2 = nn.Linear(inc2, self.midc2 * 2)

        self.v_conv = nn.Linear(inc1, 2 * self.midc1)
        self.out_conv = nn.Linear(2 * self.midc1, inc1)

        self.bn_last = nn.BatchNorm2d(inc1)
        self.dropout_p = 0.2
        self.scale = self.midc2 ** -0.5
        self._init_weight()

    def forward(self, x, x1, x2):
        B, _, H, W = x.shape
        N = H * W

        x1_norm = self.bn_x1(x1).flatten(2).transpose(1, 2)
        x2_norm = self.bn_x2(x2).flatten(2).transpose(1, 2)

        kq1 = self.kq1(x1_norm)
        kq2 = self.kq2(x2_norm)
        kq = channel_shuffle(torch.cat([kq1, kq2], dim=2), 2)
        k1, q1, k2, q2 = kq.split(self.midc2, dim=2)

        v = self.v_conv(x.flatten(2).transpose(1, 2))
        v1, v2 = v.split(self.midc1, dim=2)

        if self.need_pad:
            v1 = F.pad(v1, (0, self.pad_size))
            v2 = F.pad(v2, (0, self.pad_size))

        q1 = q1.unsqueeze(1)
        k1 = k1.unsqueeze(1)
        v1 = v1.unsqueeze(1)
        q2 = q2.unsqueeze(1)
        k2 = k2.unsqueeze(1)
        v2 = v2.unsqueeze(1)

        dropout_p = self.dropout_p if self.training else 0.0
        v1 = F.scaled_dot_product_attention(q1, k1, v1, dropout_p=dropout_p, scale=self.scale).squeeze(1)
        v2 = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=dropout_p, scale=self.scale).squeeze(1)

        if self.need_pad:
            v1 = v1[:, :, :self.midc1]
            v2 = v2[:, :, :self.midc1]

        v = torch.cat([v1, v2], dim=2)
        v = self.out_conv(v).transpose(1, 2).reshape(B, -1, H, W)
        v = self.bn_last(v) + x

        return v

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


class SpatialAttention_max(nn.Module):
    def __init__(self, in_channels, reduction1=16, reduction2=8):
        super(SpatialAttention_max, self).__init__()
        self.in_channels = in_channels
        self.scale_factor = 1.0 / (in_channels * in_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_spatial = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction1, in_channels, bias=False),
        )

        self.fc_channel = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction2, in_channels, bias=False),
        )

        self._init_weight()

    def forward(self, x):
        b, c, h, w = x.size()
        y_avg = self.avg_pool(x).squeeze(-1).squeeze(-1)

        y_spatial = self.fc_spatial(y_avg).unsqueeze(-1).unsqueeze(-1)
        y_channel = self.fc_channel(y_avg).unsqueeze(-1).unsqueeze(-1).sigmoid()

        spatial_weighted = x * y_spatial
        map = (spatial_weighted.sum(dim=1, keepdim=True) * self.scale_factor).sigmoid()
        
        return map * x * y_channel

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


class SCC_Module(nn.Module):
    def __init__(self, inc_depth2, inc_rgb):
        super(SCC_Module, self).__init__()
        channel = inc_rgb + inc_depth2

        self.fus_atten = SpatialAttention_max(in_channels=channel)
        self.conv_bn = nn.Sequential(
            nn.Conv2d(channel, inc_depth2, kernel_size=1, bias=False),
            nn.BatchNorm2d(inc_depth2)
        )

        self.cross_atten = Cross_Atten_Lite_split(inc_depth2, inc_rgb)

    def forward(self, depth_out, rgb_out):
        fus_s = self.fus_atten(torch.cat([depth_out, rgb_out], dim=1))
        fus_s = self.conv_bn(fus_s)
        fus_s = self.cross_atten(fus_s, depth_out, rgb_out)

        return fus_s


class down_sample_block(nn.Module):
    def __init__(self, inc_depth, inc_rgb, block_num, rgb_stem, depth_stem, rgb_layer, depth_layer, depth_norm):
        super(down_sample_block, self).__init__()
        self.block_num = block_num

        if block_num != 0:
            self.depth_stem = depth_stem
            self.rgb_stem = rgb_stem
        else:
            self.depth_stem = OverlapPatchEmbed(in_chans=1, embed_dim=inc_depth)
            self.rgb_stem = rgb_stem

        self.rgb_layer = rgb_layer
        self.depth_layer = depth_layer

        self.depth_norm = depth_norm

        if self.block_num != 0:
            self.SCC = SCC_Module(inc_depth2=inc_depth, inc_rgb=inc_rgb)

    def forward(self, image, depth):
        B = image.shape[0]
        image = self.rgb_stem(image)
        rgb_out = self.rgb_layer(image)

        depth_out, H, W = self.depth_stem(depth)

        for blk in self.depth_layer:
            depth_out = blk(depth_out, H, W)
        depth_out = self.depth_norm(depth_out)
        depth_out = depth_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # SCC_Ablation
        if self.block_num != 0:
            merge = self.SCC(depth_out, rgb_out)
            return rgb_out, merge
        else:
            return rgb_out, depth_out


class B0_T(nn.Module):
    def __init__(self, num_classes):
        super(B0_T, self).__init__()

        model1 = convnext_tiny(pretrained=True, drop_path_rate=0.3)
        ft1 = model1.stages
        stem = model1.downsample_layers
        stem1 = [stem[0], stem[1], stem[2], stem[3]]
        layers1 = [
            ft1[0],
            ft1[1],
            ft1[2],
            ft1[3]]

        model2 = mit_b0()
        layers2 = [model2.block1, model2.block2, model2.block3, model2.block4]
        stem2 = [model2.patch_embed1, model2.patch_embed2, model2.patch_embed3, model2.patch_embed4]
        norm2 = [model2.norm1, model2.norm2, model2.norm3, model2.norm4]

        self.channel = [32, 64, 160, 256]
        channel_list2 = [96, 192, 384, 768]

        self.down_sample_1 = down_sample_block(inc_depth=self.channel[0], inc_rgb=channel_list2[0], block_num=0,
                                               rgb_stem=stem1[0], depth_stem=None, rgb_layer=layers1[0], depth_layer=layers2[0], depth_norm=norm2[0])

        self.down_sample_2 = down_sample_block(inc_depth=self.channel[1],
                                               inc_rgb=channel_list2[1], block_num=1,
                                               rgb_stem=stem1[1], depth_stem=stem2[1], rgb_layer=layers1[1], depth_layer=layers2[1], depth_norm=norm2[1])

        self.down_sample_3 = down_sample_block(inc_depth=self.channel[2],
                                               inc_rgb=channel_list2[2], block_num=2,
                                               rgb_stem=stem1[2], depth_stem=stem2[2], rgb_layer=layers1[2], depth_layer=layers2[2], depth_norm=norm2[2])

        self.down_sample_4 = down_sample_block(inc_depth=self.channel[3],
                                               inc_rgb=channel_list2[3], block_num=3,
                                               rgb_stem=stem1[3], depth_stem=stem2[3], rgb_layer=layers1[3], depth_layer=layers2[3], depth_norm=norm2[3])

        self.Decoder = DecoderHead(in_channels=self.channel, num_classes=num_classes, dropout_ratio=0.1,
                                   norm_layer=nn.BatchNorm2d,
                                   embed_dim=256)

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict with backward compatibility for old checkpoint format.
        Converts old SCC module keys (conv1, bn) to new format (conv_bn.0, conv_bn.1).
        """
        new_state_dict = {}
        
        for key, value in state_dict.items():
            new_key = key
            
            # Convert old SCC module keys to new format
            # Old: down_sample_X.SCC.conv1.weight -> New: down_sample_X.SCC.conv_bn.0.weight
            # Old: down_sample_X.SCC.bn.* -> New: down_sample_X.SCC.conv_bn.1.*
            if '.SCC.conv1.' in key:
                new_key = key.replace('.SCC.conv1.', '.SCC.conv_bn.0.')
            elif '.SCC.bn.' in key:
                new_key = key.replace('.SCC.bn.', '.SCC.conv_bn.1.')
            
            new_state_dict[new_key] = value
        
        return super(B0_T, self).load_state_dict(new_state_dict, strict=strict)

    def forward(self, image, depth,mask, ufzs):
        input_shape = image.shape[-2:]

        rgb_out, depth_out1 = self.down_sample_1(image, depth)
        rgb_out, depth_out2 = self.down_sample_2(rgb_out, depth_out1)

        rgb_out, depth_out3 = self.down_sample_3(rgb_out, depth_out2)
        _, depth_out = self.down_sample_4(rgb_out, depth_out3)

        rgb_out = self.Decoder(
            [depth_out1,
             depth_out2,
             depth_out3,
             depth_out])
        rgb_out = F.interpolate(rgb_out, size=input_shape, mode='bilinear', align_corners=False)
        return rgb_out,1
