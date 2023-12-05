import sys
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import ArgumentParser
from tqdm import tqdm, trange
import torch
from gaussian_renderer import render
from scene import GaussianModel
from scene.scene_only import SceneOnly
from scene.GAC_features import PlyLoader
from scene.GACNet import GACNet
from random import randint
from utils.camera_utils import Camera
import os
import matplotlib.pyplot as plt

from segmentation_train import load_full_feature_gaussian, ClassifierMLP, render_feature_image


if __name__ == '__main__':
    sh_degree = 3 # hard coded for now
    num_features = 144 # 48 *3, hard coded for now
    cluster_size = 4096 # hard coded for now
    num_epoch = 30 # hard coded for now
    img_feature_dim = 3*3 # hard coded for now
    num_classes = 6 # hard coded for now
    save_freq = 5
    ckpt_path = f'checkpoints/"ckpt_ep{29}.pth"'

    parser = ArgumentParser(description="Running segmentation trainer")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--source_ply", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # load GAC layer
    xyz_all, feat_all = PlyLoader(args.source_ply).get_full_data()
    gacNet = GACNet().cuda()
    N = xyz_all.shape[1]

    # load feature gaussian layer
    feature_gaussian = load_full_feature_gaussian(args.source_ply, sh_degree, num_features)

    # load final MLP layer for segmentation
    classifier_mlp = ClassifierMLP(feature_img_dim=img_feature_dim, class_dim=num_classes).cuda()

    # other setups
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0] # for now assume no features == black
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # setup scene loader
    scene = SceneOnly(dataset)
    viewpoint_stack = None

    with torch.no_grad():
        # load the model from ckpt
        loaded_ckpt = torch.load(ckpt_path)
        gacNet.load_state_dict(loaded_ckpt['gac_state_dict'])
        classifier_mlp.load_state_dict(loaded_ckpt['mlp_state_dict'])
        metric = loaded_ckpt['metric']
        print(f"The metric is {metric}")
        
        gacNet.eval()
        classifier_mlp.eval()

        # at start of each epoch, first randomize the points going into GAC
        # random shuffle
        shuffle_idx = torch.randperm(xyz_all.shape[1])
        sel_idxes = [shuffle_idx[(i*cluster_size):((i+1)*cluster_size)] for i in range(xyz_all.shape[1] // cluster_size)]
        residual = xyz_all.shape[1] % cluster_size
        if residual!=0:
            last_block = xyz_all.shape[1] // cluster_size
            complemented_dat = torch.cat((shuffle_idx[(last_block*cluster_size):], shuffle_idx[:(cluster_size - residual)]))
            sel_idxes.append(complemented_dat)

        # generate segmentation feature every cycle
        
        seg_feat = torch.zeros((xyz_all.shape[1], 144), device=xyz_all.device)
        for idx in tqdm(sel_idxes, leave=False): #TODO -> improve efficiency
            seg_feat[idx, :] = gacNet(xyz_all[:, idx].unsqueeze(0), feat_all[:, idx].unsqueeze(0))

        # transfer segmentation features to FullFeatureGaussian
        feature_gaussian.set_features(seg_feat)

        # render a "feature image" from the feature gaussians

        # pick random camera to begin viewing
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        feature_img = render_feature_image(viewpoint_cam, feature_gaussian, pipe, bg)

        # apply MLP on the feature image
        pred_segs = classifier_mlp(feature_img) # num_class*H*W

        # calculate multi class loss
        gt_segs = viewpoint_cam.label.cuda().long()

        # use the mean accuracy as the evaluation metric
        pre_label = torch.argmax(pred_segs, dim=0).long() #H*W
        
        pre_image = pre_label.cpu().numpy()
        gt_segs = pre_label.cpu().numpy()
        
        # Create a figure with two subplots (side by side)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(pre_image)
        axes[0].set_title('Image 1')
        axes[1].imshow(gt_segs)
        axes[1].set_title('Image 2')
        plt.show()
            