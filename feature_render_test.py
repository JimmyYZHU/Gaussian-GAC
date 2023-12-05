"""
Test to see if we can define our own feature gaussians and forward and backwards it
"""

import torch
from gaussian_renderer import render
from scene import GaussianModel

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene2:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        print(f"\n\n {args.source_path}")

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # if self.loaded_iter:
        #     self.gaussians.load_ply(os.path.join(self.model_path,
        #                                                    "point_cloud",
        #                                                    "iteration_" + str(self.loaded_iter),
        #                                                    "point_cloud.ply"))
        # else:
        #     self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

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
                #  features_dc: torch.Tensor,
                #  features_rest: torch.Tensor,
                 scaling: torch.Tensor,
                 rotation: torch.Tensor,
                 opacity: torch.Tensor,
                 max_radii2D: torch.Tensor):
        super().__init__(sh_degree)
        self._xyz = xyz
        # self._features_dc = features_dc
        # self._features_rest = features_rest
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


if __name__ == '__main__':

    print("Running simple feature gaussian to test that we can propagate gradients.")

    from arguments import ModelParams, PipelineParams, OptimizationParams
    from scene import Scene
    from argparse import ArgumentParser
    import sys
    from random import randint
    from utils.loss_utils import l1_loss, ssim

    class SomeFeatureAddingLayer(torch.nn.Module):
        """
        Simulates some nn.Module that does something to our PointCloud
        """
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.w1 = torch.nn.Parameter(torch.randn(3, device="cuda"))
            self.w2 = torch.nn.Parameter(torch.randn(15, 3, device="cuda"))

        def forward(self, x: GaussianModel) -> torch.Tensor:
            return self.w1 * x._features_dc, self.w2 * x._features_rest

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--from_ply", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # imagine we got a gaussian model
    gaussians = GaussianModel(dataset.sh_degree)
    if args.start_checkpoint:
        (model_params, first_iter) = torch.load(args.start_checkpoint)
        gaussians.restore(model_params, opt)
    elif args.from_ply:
        gaussians.load_ply(args.from_ply)

    # init a base class for features
    # note that these features remain same and is used for rendering
    part_feature_gaussian = PartFeatureGaussian(
        sh_degree = dataset.sh_degree,
        xyz = gaussians._xyz,
        scaling = gaussians._scaling,
        rotation = gaussians._rotation,
        opacity = gaussians._opacity,
        max_radii2D = gaussians.max_radii2D
    )

    # set up training (need to consider how to change the values and reaccumulate gradients)
    # part_feature_gaussian.training_setup(opt) # cannot because properties are not leaves

    # feature adder
    layer = SomeFeatureAddingLayer()
    optim = torch.optim.Adam(layer.parameters())

    # other setups
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # setup scene
    scene = Scene2(dataset, part_feature_gaussian)

    ### Some training epoch

    # convert features from gaussians and set part features

    features_dc, features_rest = layer(gaussians)
    part_feature_gaussian.set_features(features_dc, features_rest)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

    render_pkg = render(viewpoint_cam, part_feature_gaussian, pipe, bg)
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image) ## to change to cross entropy loss
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

    # test our backwards, should flow to lower layers
    loss.backward()
    optim.step()
    print(f'SomeFeatureAddingLayer grads: w1={layer.w1.grad} w2={layer.w2.grad}') # not None