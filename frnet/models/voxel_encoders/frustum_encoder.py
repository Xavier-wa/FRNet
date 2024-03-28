from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch_scatter
from mmcv.cnn import build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType


@MODELS.register_module()
class FrustumFeatureEncoder(nn.Module):
    """Frustum Feature Encoder.

    Args:
        in_channels (int): Number of input features, either x, y, z or
            x, y, z, r. Defaults to 4.
        feat_channels (Sequence[int]): Number of features in each of the N
            FFELayers. Defaults to [].
        with_distance (bool): Whether to include Euclidean distance to points.
            Defaults to False.
        with_cluster_center (bool): Whether to include cluster center.
            Defaults to False.
        norm_cfg (dict or :obj:`ConfigDict`): Config dict of normalization
            layers. Defaults to dict(type='BN1d', eps=1e-5, momentum=0.1).
        with_pre_norm (bool): Whether to use the norm layer before input ffe
            layer. Defaults to False.
        feat_compression (int, optional): The frustum feature compression
            channels. Defaults to None.
    """

    def __init__(self,
                 in_channels: int = 4,
                 feat_channels: Sequence[int] = [],
                 with_distance: bool = False,
                 with_cluster_center: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN1d', eps=1e-5, momentum=0.1),
                 with_pre_norm: bool = False,
                 feat_compression: Optional[int] = None) -> None:
        super(FrustumFeatureEncoder, self).__init__()
        assert len(feat_channels) > 0

        if with_distance:
            in_channels += 1
        if with_cluster_center:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center

        feat_channels = [self.in_channels] + list(feat_channels)
        if with_pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.in_channels)[1]
        else:
            self.pre_norm = None

        ffe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            norm_layer = build_norm_layer(norm_cfg, out_filters)[1]
            if i == len(feat_channels) - 2:
                ffe_layers.append(nn.Linear(in_filters, out_filters))
            else:
                ffe_layers.append(
                    nn.Sequential(
                        nn.Linear(in_filters, out_filters, bias=False),
                        norm_layer, nn.ReLU(inplace=True)))
        self.ffe_layers = nn.ModuleList(ffe_layers)
        self.compression_layers = None
        if feat_compression is not None:
            self.compression_layers = nn.Sequential(
                nn.Linear(feat_channels[-1], feat_compression),
                nn.ReLU(inplace=True))

    def forward(self, voxel_dict: dict) -> dict:
        features = voxel_dict['voxels']
        coors = voxel_dict['coors']
        features_ls = [features]
        #去重 with batch_index 所以不怕样本之间干扰
        voxel_coors, inverse_map = torch.unique(
            coors, return_inverse=True, dim=0)

        if self._with_distance: #计算每个点的欧式距离
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)
        #featureslist  include pointdis pointxyzi
        # Find distance of x, y, and z from frustum center
        if self._with_cluster_center: #cluester the frustum area point feature(in the same frustum with the same index in inverse_map)
            voxel_mean = torch_scatter.scatter_mean(
                features, inverse_map, dim=0) #project_feature (reply on mean)
            points_mean = voxel_mean[inverse_map]#保留点的feature
            f_cluster = features[:, :3] - points_mean[:, :3] #cal the distances  算出每个frustum的均值feature_mean 然后 将每个点的feature减去这个feature_mean
            features_ls.append(f_cluster)

        # Combine together feature decorations  feature include (x),(y),(z),(i),(p_distance),(x,y,z-frustum_mean(i)x,y,z)  8channel
        features = torch.cat(features_ls, dim=-1)
        if self.pre_norm is not None:
            features = self.pre_norm(features)

        point_feats = [] #feature extracet
        for ffe in self.ffe_layers:
            features = ffe(features)
            point_feats.append(features)
        
        #frustum area max 256 features
        voxel_feats = torch_scatter.scatter_max(
            features, inverse_map, dim=0)[0]

        #feature reduce dim
        if self.compression_layers is not None:
            voxel_feats = self.compression_layers(voxel_feats)

        voxel_dict['voxel_feats'] = voxel_feats # frustum-features   (frustums num point x 16)
        voxel_dict['voxel_coors'] = voxel_coors # frustum-xyz
        voxel_dict['point_feats'] = point_feats # point-features (64,128,256,256)

        return voxel_dict
