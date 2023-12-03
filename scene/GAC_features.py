from GAC.GACNet import GACNet
import numpy as np
from plyfile import PlyData, PlyElement
from torch import nn
import torch


class PlyLoader():
    def __init__(self, ply_path):
        self.max_sh_degree = 3
        
        self.xyz = torch.empty(0)
        self.features_dc = torch.empty(0)
        self.features_rest = torch.empty(0)
        self.scaling = torch.empty(0)
        self.rotation = torch.empty(0)
        self.opacity = torch.empty(0)
        
        self.load_ply(ply_path)
        
    def load_ply(self, ply_path):
        plydata = PlyData.read(ply_path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.xyz = torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(False)
        self.features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False)
        self.features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(False)
        self.opacity = torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False)
        self.scaling = torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(False)
        self.rotation = torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False)

    def get_full_data(self):
        # ouput: xyz: [B, C, N], feat: [B, D, N], C=3, D56

        N, _ = self.xyz.shape
        xyz = self.xyz.clone().permute(1, 0).unsqueeze(0)
        
        # N*1*3, C=3
        feat_dc = self.features_dc.clone().permute(1, 2, 0)
        # N*(F/3)*3
        feat_rest = self.features_rest.clone().view(N, -1).permute(1, 0).unsqueeze(0)
        # N*1
        opacity = self.opacity.clone().permute(1, 0).unsqueeze(0)
        # N*3
        scaling = self.scaling.clone().permute(1, 0).unsqueeze(0)
        # N*4
        rotation = self.rotation.clone().permute(1, 0).unsqueeze(0)
        # size should be 3+45+1+3+4=56, [B, D, N]
        feat = torch.cat((feat_dc, feat_rest, opacity, scaling, rotation), dim=1)

        return xyz, feat


if __name__ == "__main__":
    ply_path = 'output/dd261d92-e/point_cloud/iteration_7000/point_cloud.ply'
    dataloader = PlyLoader(ply_path)