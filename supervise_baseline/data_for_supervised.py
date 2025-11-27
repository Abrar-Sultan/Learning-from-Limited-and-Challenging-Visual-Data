import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class IdentityTransform:
    '''
    returns the same image
    '''
    def __call__(self, x):
        return x

class CIFAR10Dataset(Dataset):
    '''
    A pytorch dataset that return image and their corrosponding label
    '''
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images) 

    def __getitem__(self, index):
        '''
        Returns simple augmented images and their corrosponding labels 
        '''
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            # tensor to PIL for transforms
            image = transforms.ToPILImage()(image)  
            image = self.transform(image)
        return image, label


# function to load the data
def get_dataloader(data_path, batch_size, train=True): 
    # .pt file with (images, labels)
    images, labels = torch.load(data_path) 

    # simple augmentation such random horzontal flip and random crop
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else IdentityTransform(),
        transforms.RandomCrop(32, padding=4) if train else IdentityTransform(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    dataset = CIFAR10Dataset(images, labels, transform=transform)
    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)
    return dataloader 



