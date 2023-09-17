import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class DataLoadingOperations:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_mnist(custom_transforms):
        dataset = datasets.MNIST(root="dataset/", transform=custom_transforms, download=True)
        return dataset

    @staticmethod
    def dataloading(dataset, bs=4, shuffle=False):
        loader = DataLoader(dataset, bs, shuffle)
        return loader

    @staticmethod
    def get_tfs():
        custom_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        return custom_transforms
