import numpy as np
import torch
from torch import nn
from model_utils.resnet import ResNet, BottleneckBlock
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

"""
@ARTICLE{Wang2026Simpler,
    author  = {Wang, Zenghui and Du, Songlin and Xiao, Guobao},
    journal = {IEEE Transactions on Image Processing}, 
    title   = {Simpler is Better: Feature Guard and Interaction for Semantic Correspondence}, 
    year    = {2026},
    volume  = {35},
    number  = {},
    pages   = {5626-5640},
    doi     = {10.1109/TIP.2026.3693144}
}

"""


class GuardFeatures(nn.Module):
    def __init__(self, feature_dims=[640, 1280, 1280, 768], projection_dim=1280, contrastive_temp=10, feat_map_dropout=0.2): 
        super().__init__()
        self.feature_dims = feature_dims
        self.projection_dim = projection_dim
        self.feat_map_dropout = feat_map_dropout

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.self_logit_scale = nn.Parameter(torch.ones([]) * np.log(contrastive_temp))
        self.mixing_weights = nn.Parameter(torch.ones(len(feature_dims)))
        
        self.bottle_sd_0 = nn.Sequential(
            *ResNet.make_stage(
                BottleneckBlock,
                num_blocks=1,
                in_channels=feature_dims[0],
                bottleneck_channels=projection_dim // 4,
                out_channels=projection_dim,
                norm="GN",
                num_norm_groups=32,
                kernel_size= [1,3,1],
            )
        )
        
        self.bottle_dino = nn.Sequential(
            *ResNet.make_stage(
                BottleneckBlock,
                num_blocks=1,
                in_channels=feature_dims[3],
                bottleneck_channels=projection_dim // 4,
                out_channels=projection_dim,
                norm="GN",
                num_norm_groups=32,
                kernel_size= [1,3,1],
            )
        )

    def forward(self, batch):

        if self.feat_map_dropout > 0 and self.training:
            batch = F.dropout(batch, p=self.feat_map_dropout)
        
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights, dim=0)

        sd_feature_0 = batch[:,               : 640,           :, :]
        sd_feature_1 = batch[:, 640           : 640+1280,      :, :]
        sd_feature_2 = batch[:, 640+1280      : 640+1280+1280, :, :]
        dino_feature = batch[:, 640+1280+1280 : 640+1280+1280+768, :, :]

        # CUS
        if self.projection_dim != 1280:
            if self.projection_dim < 1280:
                indices = torch.linspace(0, 1280 - 1, steps=self.projection_dim, dtype=torch.long)
                sd_feature_1 = sd_feature_1[:, indices, :, :]
                sd_feature_2 = sd_feature_2[:, indices, :, :]
            elif self.projection_dim > 1280:
                indices = torch.linspace(0, 1280 - 1, steps=self.projection_dim - 1280, dtype=torch.long)
                sd_feature_1 = torch.concat((sd_feature_1, sd_feature_1[:, indices, :, :]), dim=1)
                sd_feature_2 = torch.concat((sd_feature_2, sd_feature_2[:, indices, :, :]), dim=1)
        
        
        sd_feature_0 = self.bottle_sd_0(sd_feature_0)
        dino_feature = self.bottle_dino(dino_feature)

        output_feature = sd_feature_0 * mixing_weights[0] + sd_feature_1 * mixing_weights[1] + sd_feature_2 * mixing_weights[2] + dino_feature * mixing_weights[3] 
  
        return output_feature


class NABI(nn.Module):
    def __init__(self, projection_dim=1280):
        super().__init__()
        self.projection_dim = projection_dim

        self.source_bottlenet = nn.Sequential(
            *ResNet.make_stage(
                BottleneckBlock,
                num_blocks=1,
                in_channels=projection_dim, 
                bottleneck_channels=projection_dim // 4,
                out_channels=projection_dim,
                norm="GN",
                num_norm_groups=32,
                kernel_size= [1,3,1],
            )
        )
        
        self.target_bottlenet = nn.Sequential(
            *ResNet.make_stage(
                BottleneckBlock,
                num_blocks=1,
                in_channels=projection_dim,
                bottleneck_channels=projection_dim // 4, 
                out_channels=projection_dim,
                norm="GN",
                num_norm_groups=32,
                kernel_size= [1,3,1],
            )
        )

    def forward(self, fusion_1, fusion_2):

        agg_1 = self.source_bottlenet(fusion_1)
        agg_2 = self.target_bottlenet(fusion_2)
        int_1 = self.target_bottlenet(agg_1)
        int_2 = self.source_bottlenet(agg_2)

        perc_1 =  int_1 + agg_1 
        perc_2 =  int_2 + agg_2

        return perc_1, perc_2



class SimBetter(nn.Module):
    def __init__(self, feature_dims=[640,1280,1280,768], projection_dim=1280, feat_map_dropout=0.2, img_size=60):
        super().__init__()
        self.img_size = img_size
   
        self.proj_net = GuardFeatures(feature_dims=feature_dims, projection_dim=projection_dim, feat_map_dropout=feat_map_dropout)

        self.NABI = NABI(projection_dim=projection_dim)
                
        self.weight =  nn.Parameter(torch.tensor(0.5, dtype=torch.float), requires_grad=True) 
        
        self.apply(self._init_weights)
        # count number of parameters
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(f"SimBetter has {num_params/1e6} M parameters.")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def normalize_feats(self, feats, epsilon=1e-10):
        # (b, w*h, c)
        norms = torch.linalg.norm(feats, dim=-1)[:, :, None]
        norm_feats = feats / (norms + epsilon)
        return norm_feats

    def forward(self, feature_1, feature_2): 
        # GuFT
        fusion_1 = self.proj_net(feature_1)
        fusion_2 = self.proj_net(feature_2)

        # NABI
        perc_1, perc_2 = self.NABI(fusion_1, fusion_2)

        # calculate correlation map
        fusion_1 = fusion_1.reshape(1, 1, -1, self.img_size**2).permute(0, 1, 3, 2) 
        fusion_2 = fusion_2.reshape(1, 1, -1, self.img_size**2).permute(0, 1, 3, 2) 
        perc_1 = perc_1.reshape(1, 1, -1, self.img_size**2).permute(0, 1, 3, 2) 
        perc_2 = perc_2.reshape(1, 1, -1, self.img_size**2).permute(0, 1, 3, 2) 
        
        fusion_1 = self.normalize_feats(fusion_1[0])
        fusion_2 = self.normalize_feats(fusion_2[0])
        perc_1 = self.normalize_feats(perc_1[0]) 
        perc_2 = self.normalize_feats(perc_2[0])
        
        corr_fusion = torch.matmul(fusion_1, fusion_2.transpose(1, 2)).unsqueeze(1)
        corr_perc = torch.matmul(perc_1, perc_2.transpose(1, 2)).unsqueeze(1)
        corr_weight = torch.sigmoid(self.weight) 
        corr_final = corr_weight * corr_perc + (1 - corr_weight) * corr_fusion
        
        return fusion_1, fusion_2, corr_final

