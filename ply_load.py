from scene import GaussianModel


def test_feature(gaussians, path):
    gaussians.load_ply(path)
    
    a = 1
    