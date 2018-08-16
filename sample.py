import torch
import utils
from torchvision import datasets, transforms

data_dir = 'data/tomo'
compose = utils.define_compose(3,256)
dataset = utils.ImageFolderWithPaths(data_dir, transform=compose)
dataloader = torch.utils.data.DataLoader(dataset)

for im, label, path in dataloader:
    print(path[0])
