import torch
import torch.nn.functional as F
import numpy as np
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
import torch.nn as nn
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet3d.ops import bev_pool
from mmcv.runner import force_fp32, auto_fp16

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor(
        [int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class UNet(nn.Module):
    """
    U-Net for extracting features from satellite images.
    
    Architecture:
    - Encoder: Downsampling path with increasing channels
    - Decoder: Upsampling path with skip connections
    - Output: Feature maps aligned with BEV grid
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB)
        base_channels (int): Base number of channels in first layer
        out_channels (int): Number of output feature channels
        num_layers (int): Number of encoder/decoder layers
        bilinear (bool): Use bilinear upsampling instead of transposed conv
    """
    
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        out_channels=256,
        num_layers=4,
        bilinear=True
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bilinear = bilinear
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(in_channels, base_channels)
        
        self.down_layers = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_layers):
            out_ch = base_channels * (2 ** (i + 1))
            self.down_layers.append(Down(in_ch, out_ch))
            in_ch = out_ch
        
        # Decoder (upsampling path)
        self.up_layers = nn.ModuleList()
        for i in range(num_layers):
            in_ch = base_channels * (2 ** (num_layers - i))
            out_ch = base_channels * (2 ** (num_layers - i - 1))
            self.up_layers.append(Up(in_ch, out_ch, bilinear))
        
        # Output convolution
        self.outc = OutConv(base_channels, out_channels)
        
    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x (Tensor): Input satellite images (B, C, H, W)
        
        Returns:
            Tensor: Extracted features (B, out_channels, H, W)
        """
        # Encoder with skip connections
        skips = []
        x = self.inc(x)
        skips.append(x)
        
        for down in self.down_layers:
            x = down(x)
            skips.append(x)
        
        # Remove last skip (bottleneck)
        skips = skips[:-1]
        
        # Decoder with skip connections
        for i, up in enumerate(self.up_layers):
            skip = skips[-(i + 1)]
            x = up(x, skip)
        
        # Output
        x = self.outc(x)
        return x


class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool2d -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block: Upsample -> Concat -> DoubleConv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        Args:
            in_channels: Number of channels from decoder (before upsampling)
            out_channels: Number of output channels (also size of skip connection)
            bilinear: Use bilinear upsampling vs transposed conv
        """
        super().__init__()
        
        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After concat: in_channels (from decoder) + out_channels (from skip) -> out_channels
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            # Upsample and reduce channels at the same time
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
            # After concat: out_channels (upsampled) + out_channels (skip) -> out_channels
            self.conv = DoubleConv(out_channels * 2, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder path (to be upsampled)
            x2: Skip connection from encoder path
        """
        x1 = self.up(x1)
        
        # Handle size mismatch due to odd dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution: 1x1 conv"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BaseTransform(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        feat_down_sample,
        pc_range,
        voxel_size,
        dbound,
    ):
        super(BaseTransform, self).__init__()
        self.in_channels = in_channels
        self.feat_down_sample = feat_down_sample
        # self.image_size = image_size
        # self.feature_size = feature_size
        self.xbound = [pc_range[0],pc_range[3], voxel_size[0]]
        self.ybound = [pc_range[1],pc_range[4], voxel_size[1]]
        self.zbound = [pc_range[2],pc_range[5], voxel_size[2]]
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = None
        self.D = int((dbound[1] - dbound[0]) / dbound[2])
        # self.frustum = self.create_frustum()
        # self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self,fH,fW,img_metas):
        # iH, iW = self.image_size
        # fH, fW = self.feature_size
        iH = img_metas[0]['img_shape'][0][0]
        iW = img_metas[0]['img_shape'][0][1]
        assert iH // self.feat_down_sample == fH
        # import pdb;pdb.set_trace()
        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        # return nn.Parameter(frustum, requires_grad=False)
        return frustum
    @force_fp32()
    def get_geometry_v1(
        self,
        fH,
        fW,
        rots,
        trans,
        intrins,
        post_rots,
        post_trans,
        lidar2ego_rots,
        lidar2ego_trans,
        img_metas,
        **kwargs,
    ):
        B, N, _ = trans.shape
        device = trans.device
        if self.frustum == None:
            self.frustum = self.create_frustum(fH,fW,img_metas)
            self.frustum = self.frustum.to(device)
            # self.D = self.frustum.shape[0]
        
        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # cam_to_ego
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        # ego_to_lidar
        points -= lidar2ego_trans.view(B, 1, 1, 1, 1, 3)
        points = (
            torch.inverse(lidar2ego_rots)
            .view(B, 1, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    @force_fp32()
    def get_geometry(
        self,
        fH,
        fW,
        lidar2img,
        img_metas,
    ):
        B, N, _, _ = lidar2img.shape
        device = lidar2img.device
        # import pdb;pdb.set_trace()
        if self.frustum == None:
            self.frustum = self.create_frustum(fH,fW,img_metas)
            self.frustum = self.frustum.to(device)
            # self.D = self.frustum.shape[0]
        
        points = self.frustum.view(1,1,self.D, fH, fW, 3) \
                 .repeat(B,N,1,1,1,1)
        lidar2img = lidar2img.view(B,N,1,1,1,4,4)
        # img2lidar = torch.inverse(lidar2img)
        points = torch.cat(
            (points, torch.ones_like(points[..., :1])), -1)
        points = torch.linalg.solve(lidar2img.to(torch.float32), 
                                    points.unsqueeze(-1).to(torch.float32)).squeeze(-1)
        # points = torch.matmul(img2lidar.to(torch.float32),
        #                       points.unsqueeze(-1).to(torch.float32)).squeeze(-1)
        # import pdb;pdb.set_trace()
        eps = 1e-5
        points = points[..., 0:3] / torch.maximum(
            points[..., 3:4], torch.ones_like(points[..., 3:4]) * eps)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @force_fp32()
    def forward(
        self,
        images,
        img_metas
    ):
        B, N, C, fH, fW = images.shape
        lidar2img = []
        camera2ego = []
        camera_intrinsics = []
        img_aug_matrix = []
        lidar2ego = []

        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
            camera2ego.append(img_meta['camera2ego'])
            camera_intrinsics.append(img_meta['camera_intrinsics'])
            img_aug_matrix.append(img_meta['img_aug_matrix'])
            lidar2ego.append(img_meta['lidar2ego'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = images.new_tensor(lidar2img)  # (B, N, 4, 4)
        camera2ego = np.asarray(camera2ego)
        camera2ego = images.new_tensor(camera2ego)  # (B, N, 4, 4)
        camera_intrinsics = np.asarray(camera_intrinsics)
        camera_intrinsics = images.new_tensor(camera_intrinsics)  # (B, N, 4, 4)
        img_aug_matrix = np.asarray(img_aug_matrix)
        img_aug_matrix = images.new_tensor(img_aug_matrix)  # (B, N, 4, 4)
        lidar2ego = np.asarray(lidar2ego)
        lidar2ego = images.new_tensor(lidar2ego)  # (B, N, 4, 4)

        # import pdb;pdb.set_trace()
        # lidar2cam = torch.linalg.solve(camera2ego, lidar2ego.view(B,1,4,4).repeat(1,N,1,1))
        # lidar2oriimg = torch.matmul(camera_intrinsics,lidar2cam)
        # mylidar2img = torch.matmul(img_aug_matrix,lidar2oriimg)



        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]

        # tmpgeom = self.get_geometry(
        #     fH,
        #     fW,
        #     mylidar2img,
        #     img_metas,
        # )

        geom = self.get_geometry_v1(
            fH,
            fW,
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            lidar2ego_rots,
            lidar2ego_trans,
            img_metas
        )
        x = self.get_cam_feats(images)
        x = self.bev_pool(geom, x)
        # import pdb;pdb.set_trace()
        x = x.permute(0,1,3,2).contiguous()
        
        return x



@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LSSTransform(BaseTransform):
    def __init__(
        self,
        in_channels,
        out_channels,
        feat_down_sample,
        pc_range,
        voxel_size,
        dbound,
        downsample=1,
    ):
        super(LSSTransform, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feat_down_sample=feat_down_sample,
            pc_range=pc_range,
            voxel_size=voxel_size,
            dbound=dbound,
        )
        # import pdb;pdb.set_trace()
        self.depthnet = nn.Conv2d(in_channels, int(self.D + self.C), 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, images, img_metas):
        x = super().forward(images, img_metas)
        x = self.downsample(x)
        return x