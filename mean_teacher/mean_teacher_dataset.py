import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from mean_teacher_transforms import MeanTeacherAugmentation


NO_LABEL = -1

class CIFAR10LabeledDataset(Dataset): 
    '''
    A pytorch dataset for labeled training data where each image is augmented twice and 
    returns a tuple containing two augmented images and the corresponding ground-truth label
    '''
    def __init__(self, images, labels, transform=None): 
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        '''
        Returns two augmented images and the corresponding label ((img_aug1, img_aug2), label)
        '''
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            # convert tensors to PIL image for transformation
            image = transforms.ToPILImage()(image)  
            image_aug_1, image_aug_2 = self.transform(image)
        return (image_aug_1, image_aug_2), label
    
    

class CIFAR10UnlabeledDataset(Dataset):
     '''
     A pytorch dataset for unlabeled training data where each image is augmented twice and 
     returns a tuple containing two augmented images 
     and -1 to indicate no label for the image
     '''
     def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

     def __len__(self):
        return len(self.images)

     def __getitem__(self, index):
         '''
        Returns two augmented images ((img_aug1, img_aug2), -1)
        '''
         image = self.images[index]
         if self.transform:
            # convert tensors to PIL image for transformation
            image = transforms.ToPILImage()(image)  
            image_aug_1, image_aug_2 = self.transform(image)
         return (image_aug_1, image_aug_2), NO_LABEL


class CIFAR10ValDataset(Dataset):
    '''
    A pytorch dataset for labeled validation/test data
    '''
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        '''
        Returns image and the corresponding label 
        '''
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            # convert tensors to PIL image for transformation
            image = transforms.ToPILImage()(image)  
            image = self.transform(image)
        return image, label


# function to load the labeled data
def get_train_dataloader_labeled(data_path, batch_size, num_samples_label):
    # .pt file with (images, labels)
    images_l, labels_l = torch.load(data_path)

    # augmentation
    mean_teacher_transform = MeanTeacherAugmentation()
    dataset = CIFAR10LabeledDataset(images_l, labels_l, transform=mean_teacher_transform)

    # setting up random sampler so we can sample enough label data to run all training steps
    train_label_data_sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples_label)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_label_data_sampler, num_workers=4, pin_memory=False, drop_last=False)

    return dataloader


def get_train_dataloader_unlabeled(data_path, batch_size, num_samples_unlabel):
    # .pt file with (images, labels)
    images_u, labels_u = torch.load(data_path)

    # augmentation
    mean_teacher_transform = MeanTeacherAugmentation()
    dataset = CIFAR10UnlabeledDataset(images_u, transform=mean_teacher_transform)

    # setting up random sampler so we can sample enough data to run all training steps
    train_unlabel_data_sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples_unlabel)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_unlabel_data_sampler, num_workers=4, pin_memory=False, drop_last=False)

    return dataloader


def get_val_dataloader_labeled(data_path, batch_size):
    # .pt file with (images, labels)
    images_l, labels_l = torch.load(data_path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
    ])

    dataset = CIFAR10ValDataset(images_l, labels_l, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return dataloader