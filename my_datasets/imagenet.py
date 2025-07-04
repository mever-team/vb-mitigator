import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

class ImageNet(datasets.ImageFolder):
    def __getitem__(self, index):
        # Get original tuple: (image, class)
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Modify this part as needed, e.g., add index or path to the return
        return {"inputs": sample, "targets": target, "unknown": target, "index": index}