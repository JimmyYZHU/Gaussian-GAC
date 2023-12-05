"""
Test to see if we can define our own feature gaussians and forward and backwards it
"""

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

class PartFeatureGaussian(GaussianModel):
    """
    Part of the Feature Gaussians. Rapid prototype of feature Gaussians using original Gaussian Splat
    classes.

    Takes in feature-ful Gaussians from GAC, convert them back into
    Gaussian Model for rendering

    - Simple implementation that just output 3 features per 48 features
    - As compared to a full sphereical harmonics function, we render 3 channels at a time,
        the idea is similar to a depth wise convulation only does mixing between certain channels
        and a seperate extraction for depthwise features
    """
    def __init__(self, sh_degree: int,
                 xyz: torch.Tensor,
                 scaling: torch.Tensor,
                 rotation: torch.Tensor,
                 opacity: torch.Tensor,
                 max_radii2D: torch.Tensor):
        super().__init__(sh_degree)
        self._xyz = xyz
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity
        self.max_radii2D = max_radii2D

    def set_features(self, features_dc: torch.Tensor, features_rest: torch.Tensor):
        """
        Set features without influencing gradients
        """
        self._features_dc = features_dc
        self._features_rest = features_rest

class FullFeatureGaussian:
    """
    Represents the feature gaussian as K PartFeatureGaussians to reuse available code
    """
    # constants
    dc_size = 3
    rest_row = 15
    rest_col = 3
    split_size = dc_size + rest_row * rest_col # for now everything are in multiples of 48

    def __init__(self, feature_size: int,
                 sh_degree: int,
                 xyz: torch.Tensor,
                 scaling: torch.Tensor,
                 rotation: torch.Tensor,
                 opacity: torch.Tensor,
                 max_radii2D: torch.Tensor):
        assert feature_size % FullFeatureGaussian.split_size == 0, "Feature size must be in multiples of 15 (for now)"
        self.K = int(feature_size / FullFeatureGaussian.split_size)
        self._xyz = xyz
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity
        self.max_radii2D = max_radii2D
        self.part_features = [PartFeatureGaussian(sh_degree, xyz, scaling, rotation, opacity, max_radii2D)
                              for _ in range(self.K)]

    def set_features(self, features: torch.Tensor) -> None:
        """
        Sets the features into the individual PartFeatureGaussian in preperation for rendering step

        Inputs:
        - features: (N, feature_size) tensor
        """
        for i, part_feature in enumerate(self.part_features):
            partial_sh_features = features[:, i*FullFeatureGaussian.split_size: (i+1)*FullFeatureGaussian.split_size]
            features_dc = partial_sh_features[:, None, :FullFeatureGaussian.dc_size]
            features_rest = partial_sh_features[:, FullFeatureGaussian.dc_size:]
            features_rest = features_rest.reshape(-1, FullFeatureGaussian.rest_row, FullFeatureGaussian.rest_col)
            part_feature.set_features(features_dc, features_rest)

def load_full_feature_gaussian(ply_file_path: str, sh_degree: int, feature_size: int) -> FullFeatureGaussian:
    """
    Handles construction of FullFeatureGaussian.
    - Also hides ugly instantiation of temporary GaussianModel object
    - TODO: refactor in the future
    """
    # load a temp gaussian object which hopefully gets garbage collected early
    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_ply(ply_file_path)

    # create FullFeatureGaussian, basically we are retaining everything except the colour features
    return FullFeatureGaussian(feature_size = feature_size,
                               sh_degree = dataset.sh_degree,
                               xyz = gaussians._xyz.cuda(),
                               scaling = gaussians._scaling.cuda(),
                               rotation = gaussians._rotation.cuda(),
                               opacity = gaussians._opacity.cuda(),
                               max_radii2D = gaussians.max_radii2D.cuda())

class ClassifierMLP(torch.nn.Module):
    """
    Final classifier MLP that generates segmentation labels

    Use MLP along the the feature dimension of each pixel output (using 2D conv with 1 kernel)
    """
    def __init__(self,
                 feature_img_dim: int, class_dim: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torch.nn.Sequential(torch.nn.Conv2d(feature_img_dim, 32, 1, 1),
                                         torch.nn.Dropout(0.3),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(32, 64, 1, 1),
                                         torch.nn.Dropout(0.3),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, class_dim, 1, 1))

    def forward(self, x: torch.Tensor):
        """
        Inputs:
        - x: (feature_img_dim, H, W) tensor
        Returns:
        - pred: (class_dim, H, W) tensor
        """
        return self.model(x)

def render_feature_image(viewpoint_cam: Camera,
                         full_feature_gaussian: FullFeatureGaussian,
                         pipe: PipelineParams,
                         bg: torch.Tensor) -> torch.Tensor:
    """
    Converts full_feature gaussians into feature images

    Input:

    - full_feature_gaussian: FullFeatureGaussian to render
    Returns:
    - feature_image: (F,H,W) tensor, F = feature_size/48 * 3
    """
    # each image C,H,W
    imgs = [render(viewpoint_cam, part_feature_gaussian, pipe, bg)["render"]
            for part_feature_gaussian in full_feature_gaussian.part_features]
    return torch.concat(imgs, dim=-3) # concat on C dimension

if __name__ == '__main__':
    sh_degree = 3 # hard coded for now
    num_features = 144 # 48 *3, hard coded for now
    cluster_size = 4096 # hard coded for now
    num_epoch = 30 # hard coded for now
    img_feature_dim = 3*3 # hard coded for now
    num_classes = 6 # hard coded for now
    save_freq = 5
    ckpt_path = 'checkpoints/klevr'

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

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

    # setup optimizers
    param_list = list(gacNet.parameters()) + list(classifier_mlp.parameters())
    optim = torch.optim.Adam(param_list,
                                    lr=0.01,
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    weight_decay=1e-4)
    # optim_classifier = torch.optim.Adam(classifier_mlp.parameters(),
    #                                     lr=0.005,
    #                                     weight_decay=0.01)

    # setup criterion
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0], device="cuda"))

    # precalculations for shuffle ops
    quot, rem = divmod(N, cluster_size)

    # training cycle
    losses = []
    metrics = []
    pbar = trange(num_epoch)
    for epoch in pbar:

        # set to train and clear all gradients
        gacNet.train()
        optim.zero_grad()

        # firstly, randomize the points going into GAC
        # random shuffle
        shuffle_idx = torch.randperm(N) if rem == 0 else \
                        torch.concat((torch.randperm(N),torch.randperm(N)))[:((quot+1)*cluster_size)]
        shuffle_idx = shuffle_idx.view(-1, cluster_size)
        features = gacNet(xyz_all[:, shuffle_idx].transpose(-2,-3), feat_all[:, shuffle_idx].transpose(-2,-3))
        seg_feat = torch.zeros((N, 144), device=xyz_all.device)
        for i,idx in enumerate(shuffle_idx): #TODO -> improve efficiency
            seg_feat[idx, :] = features[i,...]

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
        loss = criterion(pred_segs[None, ...], gt_segs[None, ...])

        # backprop and update weights
        loss.backward()
        optim.step()

        # use the mean accuracy as the evaluation metric
        pre_label = torch.argmax(pred_segs, dim=0).long() #H*W

        class_acc = - torch.ones(num_classes-1) # exclude background
        for label in range(num_classes):
            if label==0:
                continue
            gt_num = torch.sum(gt_segs==label)
            if gt_num>0:
                class_acc[label] = torch.sum((pre_label==label) & (gt_segs==label)) / gt_num

        metric = torch.mean(class_acc[class_acc!=-1])

        # print out current loss for sanity check
        pbar.set_description(f"epoch {epoch} loss: {loss.item()} metric: {metric.item()}")

        # store losses and metrics for debugging
        losses.append(loss.item())
        metrics.append(metric.item())

        #TODO: set checkpoints and weights
        if (epoch+1) % save_freq == 0:
            ckpt = {
                'epoch': epoch+1,
                'gac_state_dict': gacNet.state_dict(),
                'mlp_state_dict': classifier_mlp.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': losses,
                'metric': metrics
            }
            torch.save(ckpt, os.path.join(ckpt_path, f'ckpt_ep{epoch}.pth'))
