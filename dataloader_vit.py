from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


train_transforms = A.Compose([
    A.PadIfNeeded(min_height=40, min_width=40, p=0.5),
    A.RandomCrop(height=32, width=32, p=1),
    A.HorizontalFlip(),
    A.ShiftScaleRotate(),
    A.CoarseDropout(max_holes=1, min_height=8, min_width=8, max_height=8, max_width=8, fill_value= [0.4914*255, 0.4822*255, 0.4465*255]),
    A.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ToTensorV2(),  
])


test_transforms = A.Compose([
    A.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
    ToTensorV2(),  
])


class cifar_class(Dataset):
        def __init__(self, data, transforms):
            super().__init__()
            self.data = data
            self.transforms = transforms
        def __len__(self):
            return len(self.data)
        def __getitem__(self, index):
            image = self.data[index][0]
            label = self.data[index][1]
            transformed_image = self.transforms(image=np.array(image))["image"]
            return transformed_image, label

def get_dataloader():
    train_data = datasets.CIFAR10("./data",train=True, transform=None, download=True)
    test_data = datasets.CIFAR10("./data",train=False, transform=None, download=True)

    train_data_aug = cifar_class(train_data,train_transforms)
    test_data_aug = cifar_class(test_data,test_transforms)

    train_loader = DataLoader(train_data_aug, batch_size=32, shuffle=True, num_workers=5, persistent_workers=True)
    test_loader = DataLoader(test_data_aug, batch_size=32, shuffle=False, num_workers=2, persistent_workers=True)
    return train_loader, test_loader

