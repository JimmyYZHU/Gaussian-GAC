from scene import GaussianModel
from GAC.GACNet import GACNet



def test_feature(gaussians, path):
    gaussians.load_ply(path)
    num_class = 4
    GAC_model = GACNet(num_class)
    
    a = 1
    