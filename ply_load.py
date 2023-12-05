from scene import GaussianModel
from torch.utils.data import Dataset
from scene.GACNet import GACNet


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
    GAC_model = GACNet()
    