from GACNet import GACNet
import numpy as np
from plyfile import PlyData, PlyElement
from torch import nn
import torch
from tqdm import tqdm, trange
import argparse

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
        # N = 4096
        
        xyz = self.xyz.permute(1, 0)
        
        # N*1*3, C=3
        feat_dc = self.features_dc.permute(1, 2, 0)
        feat_dc = feat_dc.squeeze()
        # N*(F/3)*3
        feat_rest = self.features_rest.view(N, -1).permute(1, 0)
        # N*1
        opacity = self.opacity.permute(1, 0)
        # N*3
        scaling = self.scaling.permute(1, 0)
        # N*4
        rotation = self.rotation.permute(1, 0)
        # size should be 3+45+1+3+4=56, [B, D, N]
        feat = torch.cat((feat_dc, feat_rest, opacity, scaling, rotation), dim=0)

        M = 4096
        return xyz[:, :M], feat[:, :M]


def training():
    ply_path = 'output/dd261d92-e/point_cloud/iteration_30000/point_cloud.ply'
    dataloader = PlyLoader(ply_path)
    
    model = GACNet()
    model.to('cuda')
    
    data_size = 4096
    # xyz_all: 3*N
    xyz_all, feat_all = dataloader.get_full_data()
    
    # random shuffle
    shuffle_idx = torch.randperm(xyz_all.shape[1])
    sel_idxes = [shuffle_idx[(i*data_size):((i+1)*data_size)] for i in range(xyz_all.shape[1] // data_size)]
    residual = xyz_all.shape[1] % data_size
    if residual!=0:
        last_block = xyz_all.shape[1] // data_size
        complemented_dat = torch.cat((shuffle_idx[(last_block*data_size):], shuffle_idx[:residual]))
        sel_idxes.append(complemented_dat)
    
    seg_feat = torch.zeros((1, xyz_all.shape[1], 128), device=xyz_all.device)
    
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
    
    num_epoch = 10
    criterion = nn.CrossEntropyLoss()
    
    for epoch in trange(num_epoch):
        model.train()
        optimizer.zero_grad()
        for idx in sel_idxes:
            seg_feat[:, idx, :] = model(xyz_all[:, idx].unsqueeze(0), feat_all[:, idx].unsqueeze(0))
        
        loss = criterion(seg_feat, torch.ones(seg_feat.shape, device=seg_feat.device))
        # loss.backward()
        print(f"epoch {epoch} loss: {loss}")
        optimizer.step()
    
    print(seg_feat[0, :3, :10])



if __name__ == "__main__":
    training()