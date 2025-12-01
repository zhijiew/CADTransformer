import torch
import torch.nn as nn
import torch.nn.functional as F

from models.seg_hrnet import get_seg_model
from utils.utils_model import _FEAT_DIMS
from lib.vector_encoder import VectorEncoder, VectorHead


class DualBranchCADFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_class + 1
        self.backbone = get_seg_model(cfg)

        backbone_channels = sum(_FEAT_DIMS[cfg.MODEL.BACKBONE])
        self.seg_head = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_channels, self.num_classes, kernel_size=1)
        )

        vector_in_dim = getattr(cfg, "vector_in_dim", 128)
        vector_hidden_dim = getattr(cfg, "vector_hidden_dim", 128)
        vector_out_dim = getattr(cfg, "vector_out_dim", 256)
        vector_num_layers = getattr(cfg, "vector_num_layers", 2)
        align_dim = getattr(cfg, "align_dim", vector_out_dim)

        self.vector_encoder = VectorEncoder(vector_in_dim, vector_hidden_dim, vector_out_dim, num_layers=vector_num_layers)
        self.vector_head = VectorHead(vector_out_dim, self.num_classes)

        self.raster_proj = nn.Linear(backbone_channels, align_dim)
        self.vector_proj = nn.Linear(vector_out_dim, align_dim)

    def _fuse_backbone_features(self, features):
        if not isinstance(features, (list, tuple)):
            return features
        x0 = features[0]
        x0_h, x0_w = x0.size(2), x0.size(3)
        resized = [x0]
        for feat in features[1:]:
            resized.append(F.interpolate(feat, size=(x0_h, x0_w), mode='bilinear', align_corners=True))
        fused = torch.cat(resized, dim=1)
        return fused

    def forward(self, image, vec_x=None, vec_edge_index=None, **kwargs):
        backbone_feats = self.backbone(image)
        fused = self._fuse_backbone_features(backbone_feats)
        raster_logits = self.seg_head(fused)

        raster_global = fused.mean(dim=(2, 3))
        rast_z = F.normalize(self.raster_proj(raster_global), dim=-1)

        vector_logits_list = []
        pooled_vectors = []
        if vec_x is not None and vec_edge_index is not None:
            for x_item, edge_item in zip(vec_x, vec_edge_index):
                node_feat = self.vector_encoder(x_item, edge_item)
                vector_logits_list.append(self.vector_head(node_feat))
                pooled_vectors.append(node_feat.mean(dim=0))

        if pooled_vectors:
            pooled_vectors = torch.stack(pooled_vectors, dim=0)
            vect_z = F.normalize(self.vector_proj(pooled_vectors), dim=-1)
        else:
            vect_z = None

        return {
            "raster_logits": raster_logits,
            "vector_logits": vector_logits_list,
            "rast_z": rast_z,
            "vect_z": vect_z,
        }
