# torch libraries
import torch
import torch.nn as nn
# customized libraries
from .EfficientNet import EfficientNet
import torch.nn.functional as F
class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class SoftGroupingStrategy(nn.Module):
    def __init__(self, in_channel, out_channel, N):
        super(SoftGroupingStrategy, self).__init__()
        self.g_conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[0], bias=False)
        self.g_conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[1], bias=False)
        self.g_conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=N[2], bias=False)

    def forward(self, q):
        return self.g_conv1(q) + self.g_conv2(q) + self.g_conv3(q)


class GradientInducedTransition(nn.Module):
    def __init__(self, channel, M, N):
        super(GradientInducedTransition, self).__init__()
        self.M = M

        self.downsample2 = nn.Upsample(scale_factor=1 / 2, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=1 / 4, mode='bilinear', align_corners=True)

        self.sgs3 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs4 = SoftGroupingStrategy(channel + 32, channel, N=N)
        self.sgs5 = SoftGroupingStrategy(channel + 32, channel, N=N)

    def forward(self, xr3, xr4, xr5, xg):
        # transmit the gradient cues into the context embeddings
        q3 = self.gradient_induced_feature_grouping(xr3, xg, M=self.M[0])
        q4 = self.gradient_induced_feature_grouping(xr4, self.downsample2(xg), M=self.M[1])
        q5 = self.gradient_induced_feature_grouping(xr5, self.downsample4(xg), M=self.M[2])

        # attention residual learning
        zt3 = xr3 + self.sgs3(q3)
        zt4 = xr4 + self.sgs4(q4)
        zt5 = xr5 + self.sgs5(q5)

        return zt3, zt4, zt5

    def gradient_induced_feature_grouping(self, xr, xg, M):
        if not M in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Invalid Group Number!: must be one of [1, 2, 4, 8, 16, 32]")

        if M == 1:
            return torch.cat((xr, xg), 1)

        xr_g = torch.chunk(xr, M, dim=1)
        xg_g = torch.chunk(xg, M, dim=1)
        foo = list()
        for i in range(M):
            foo.extend([xr_g[i], xg_g[i]])

        return torch.cat(foo, 1)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        # 第一个深度可分离卷积层
        self.conv1_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                         groups=in_channels, bias=False)
        self.conv1_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.SyncBatchNorm(out_channels)

        # 第二个深度可分离卷积层
        self.conv2_depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                         groups=out_channels, bias=False)
        self.conv2_pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.SyncBatchNorm(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 第一个深度可分离卷积层
        out = self.conv1_depthwise(x)
        out = self.conv1_pointwise(out)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个深度可分离卷积层
        out = self.conv2_depthwise(out)
        out = self.conv2_pointwise(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class BlockWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlockWithSkip, self).__init__()

        # Skip连接部分
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.skipbn = nn.SyncBatchNorm(out_channels)

        # 主要部分
        self.rep = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.SyncBatchNorm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.SyncBatchNorm(out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Skip连接
        skip = self.skip(x)
        skip = self.skipbn(skip)

        # 主要部分
        out = self.rep(x)

        # 合并主要部分和Skip连接
        out = out + skip
        out = self.relu(out)

        return out


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeparableConvBlock, self).__init__()

        # 第一个深度可分离卷积层
        self.conv1_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                         groups=in_channels, bias=False)
        self.conv1_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.SyncBatchNorm(out_channels)

        # 第二个深度可分离卷积层
        self.conv2_depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                         groups=out_channels, bias=False)
        self.conv2_pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.SyncBatchNorm(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 第一个深度可分离卷积层
        out = self.conv1_depthwise(x)
        out = self.conv1_pointwise(out)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个深度可分离卷积层
        out = self.conv2_depthwise(out)
        out = self.conv2_pointwise(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample2 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample3 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample4 = ConvBR(channel, channel, 3, padding=1)
        self.conv_upsample5 = ConvBR(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = ConvBR(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = ConvBR(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 3, 1)

    def forward(self, zt5, zt4, zt3):
        zt5_1 = zt5
        zt4_1 = self.conv_upsample1(self.upsample(zt5)) * zt4
        zt3_1 = self.conv_upsample2(self.upsample(zt4_1)) * self.conv_upsample3(self.upsample(zt4)) * zt3

        zt4_2 = torch.cat((zt4_1, self.conv_upsample4(self.upsample(zt5_1))), 1)
        zt4_2 = self.conv_concat2(zt4_2)

        zt3_2 = torch.cat((zt3_1, self.conv_upsample5(self.upsample(zt4_2))), 1)
        zt3_2 = self.conv_concat3(zt3_2)

        pc = self.conv4(zt3_2)
        pc = self.conv5(pc)

        return pc


class TextureEncoder(nn.Module):
    def __init__(self):
        super(TextureEncoder, self).__init__()
        self.conv1 = ConvBR(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBR(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBR(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv_out = ConvBR(32, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        xg = self.conv3(feat)
        pg = self.conv_out(xg)
        return xg, pg

class GuidedAttention(nn.Module):
    """ Reconstruction Guided Attention. """

    def __init__(self, depth=64, drop_rate=0.2):
        super(GuidedAttention, self).__init__()
        self.depth = depth
        self.gated = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.h = nn.Sequential(
            nn.Conv2d(depth, depth, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(True),
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, pred_x, embedding):
        residual_full = torch.abs(x - pred_x)
        residual_x = F.interpolate(residual_full, size=embedding.shape[-2:],
                                   mode='bilinear', align_corners=True)
        res_map = self.gated(residual_x)
        return res_map * self.h(embedding) + self.dropout(embedding)

class DGNet(nn.Module):
    def __init__(self, channel=32, arc='B0', M=[8, 8, 8], N=[4, 8, 16]):
        super(DGNet, self).__init__()

        if arc == 'EfficientNet-B1':
            print('--> using efficientnet-b1 right now')
            self.context_encoder = EfficientNet.from_pretrained('efficientnet-b1')
            in_channel_list = [40, 112, 320]
        elif arc == 'EfficientNet-B4':
            print('--> using efficientnet-b4 right now')
            self.context_encoder = EfficientNet.from_pretrained('efficientnet-b4')
            in_channel_list = [56, 160, 448]
        else:
            raise Exception("Invalid Architecture Symbol: {}".format(arc))

        self.texture_encoder = TextureEncoder()

        self.dr3 = DimensionalReduction(in_channel=in_channel_list[0], out_channel=channel)
        self.dr4 = DimensionalReduction(in_channel=in_channel_list[1], out_channel=channel)
        self.dr5 = DimensionalReduction(in_channel=in_channel_list[2], out_channel=channel)

        self.git = GradientInducedTransition(channel=channel, M=M, N=N)
        self.ncd = NeighborConnectionDecoder(channel=channel)

        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.attention = GuidedAttention(depth=64, drop_rate=0.2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=1024, out_features=2, bias=True)
        self.block1 = Block(in_channels=64,out_channels=64)
        self.block2 = BlockWithSkip(in_channels=64,out_channels=512)
        self.separableConvBlock = SeparableConvBlock(in_channels=512,out_channels=1024)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x,gradient_image):
        # context path (encoder)
        endpoints = self.context_encoder.extract_endpoints(x)
        x3 = endpoints['reduction_3']
        x4 = endpoints['reduction_4']
        x5 = endpoints['reduction_5']
        # print(x3.shape, x4.shape, x5.shape)
        xr3 = self.dr3(x3)
        xr4 = self.dr4(x4)
        xr5 = self.dr5(x5)

        # spatial path (encoder)
        xg, pg = self.texture_encoder(x)

        # decoder
        zt3, zt4, zt5 = self.git(xr3, xr4, xr5, xg)

        pc = self.ncd(zt5, zt4, zt3)
        recons_x = self.upsample(pc)
        img_att = self.attention(x, recons_x, zt5)
        embedding = self.block1(img_att)
        embedding = self.block2(embedding)
        embedding = self.separableConvBlock(embedding)
        embedding = self.global_pool(embedding).squeeze()
        out = self.dropout(embedding)
        return self.fc(out),recons_x,self.upsample(pg)


if __name__ == '__main__':
    net = DGNet(channel=64, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16]).eval()
    inputs = torch.randn(1, 3, 352, 352)
    outs = net(inputs)
    print(outs[0].shape)