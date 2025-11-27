import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from fixmatch_transforms import FixMatchAugmentation 

class LabeledDataset(Dataset):
    '''
    pytorch dataset for labeled training data where each image is weakl augmented  and 
    returns the augmented images and the corresponding ground-truth labels
    '''
    def __init__(self, images, labels, transform=None): 
        self.images = images
        self.labels = labels 
        self.transform = transform 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        '''
        Returns weakly augmented images and the corresponding labels
        '''
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            # convert tensors to PIL image for transformation
            image = transforms.ToPILImage()(image)  
            image = self.transform(image)
        return image, label
    

class UnlabeledDataset(Dataset):
     '''
     A pytorch dataset for unlabeled training data where each image go through a weak augmentation and
     strong augmentation returns two augmented images (weak_aug_image, strong_aug_image)
     '''
     def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform 

     def __len__(self):
        return len(self.images)

     def __getitem__(self, index): 
         '''
         Returns weakly augmented images and the strongly augmented image
         here, image = (weak_aug_image, strong_aug_image)
         '''
         image = self.images[index]
         if self.transform:
            # convert tensors to PIL image for transformation
            image = transforms.ToPILImage()(image)  
            image = self.transform(image)
         return image  

     
# function to load the labeled data
def get_train_dataloader_labeled(data_path, batch_size, num_samples_label):
    # .pt file with (images, labels)
    images_l, labels_l = torch.load(data_path)

    # weak augmentation
    weak_transform = FixMatchAugmentation(use_strong_aug=False)
    dataset = LabeledDataset(images_l, labels_l, transform=weak_transform)

    # setting up random sampler to have enough data to run all the training steps
    train_label_data_sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples_label)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_label_data_sampler, num_workers=4, drop_last=False)

    return dataloader 


def get_train_dataloader_unlabeled(data_path, batch_size, num_samples_label, mu=7, strong_aug=True):
    # .pt file with (images, labels)
    images_u, labels_u = torch.load(data_path)

    # strong augmentation
    strong_augmentation = FixMatchAugmentation(use_strong_aug=strong_aug)
    dataset = UnlabeledDataset(images_u, transform=strong_augmentation)

    # setting up random sampler to have enough data to run all the training steps
    train_unlabel_data_sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples_label * mu)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size * mu, sampler=train_unlabel_data_sampler, num_workers=4, drop_last=False)

    return dataloader


def get_val_dataloader_labeled(data_path, batch_size): 
    # .pt file with (images, labels)
    images_l, labels_l = torch.load(data_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)), # CIFAR 10 
        # transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)), # CIFAR 100

    ])

    dataset = LabeledDataset(images_l, labels_l, transform=transform)
    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader