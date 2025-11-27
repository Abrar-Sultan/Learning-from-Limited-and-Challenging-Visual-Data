import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from remixmatch_transforms import ReMixMatchAugmentation


class LabeledDataset(Dataset):
    '''
    pytorch dataset for labeled training data where each image is weakly augmented  and 
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
        returns weakly augmented images and the corresponding labels
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
     pytorch dataset for unlabeled training data where each image go through a weak augmentation and
     k strong augmentation returns (weak_aug_image, k strong_aug_image)
     '''
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
         image = self.images[index]
         if self.transform:
            # convert tensors to PIL image for transformation
            image = transforms.ToPILImage()(image)  
            image = self.transform(image)  
        # return tuple like: (weak, [strong...k])
         return image  
     

# function to load the labeled data
def get_train_dataloader_labeled(data_path, batch_size, total_image):
    # .pt file with (images, labels)
    images_l, labels_l = torch.load(data_path)

    # weak augmentation
    weak_transform = ReMixMatchAugmentation(use_label_img_aug=True, use_cta=False)
    dataset = LabeledDataset(images_l, labels_l, transform=weak_transform)

    # setting up random sampler to have enough data to run all the training steps
    train_label_data_sampler = RandomSampler(dataset, replacement=True, num_samples=total_image)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_label_data_sampler, num_workers=4, drop_last=True)

    return dataloader


def get_train_dataloader_unlabeled(data_path, batch_size, k, total_image, use_label_aug=False, use_cta=False, cta=None):
    # .pt file with (images, labels)
    images_u, labels_u = torch.load(data_path)

    # strong augmentation
    strong_augmentation = ReMixMatchAugmentation(k=k, use_label_img_aug=use_label_aug, use_cta=use_cta, cta=cta) 
    dataset = UnlabeledDataset(images_u, transform=strong_augmentation)

    # setting up random sampler how have enough data to run all the training steps
    train_unlabel_data_sampler = RandomSampler(dataset, replacement=True, num_samples=total_image)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_unlabel_data_sampler, num_workers=4, drop_last=True)

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


def get_cta_probe_dataloader(data_path, batch_size, cta, total_image):
    # this dataloader is only used for updating cta policy
    # .pt file with (images, labels)
    images_l, labels_l = torch.load(data_path)

    # augmentation
    cta_transform = ReMixMatchAugmentation(use_label_img_aug=False, use_cta=True, use_cta_probe=True, cta=cta)
    dataset = LabeledDataset(images_l, labels_l, transform=cta_transform)

    # setting up random sampler to have enough data to run all training steps
    data_sampler = RandomSampler(dataset, replacement=True, num_samples=total_image)

    # dataloader (returns a dictionary containing augmented image, label and applied augmentaion policy)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, num_workers=4, 
                            drop_last=False, collate_fn=cta_collate)

    return dataloader


def cta_collate(dataset):
    # convert tuples ((aug_image, policy), label) into dictionary
    image = torch.stack([data[0][0] for data in dataset], dim=0)
    label   = torch.tensor([data[1] for data in dataset], dtype=torch.long)
    policy = [data[0][1] for data in dataset] 
    return {"image": image, "target": label, "policy": policy}

     

