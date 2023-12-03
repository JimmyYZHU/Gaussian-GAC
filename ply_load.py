from scene import GaussianModel
from GAC.GACNet import GACNet
from torch.utils.data import Dataset

class S3DISDataLoader(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]



def test_feature(gaussians, path):
    gaussians.load_ply(path)
    num_class = 4
    GAC_model = GACNet(num_class)
    
    a = 1
    