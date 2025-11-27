import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def interleave_offsets(batch_size, nu):
    '''
    this code is adopted from: https://github.com/TorchSSL/TorchSSL/tree/main/models/remixmatch
    '''
    # nu = number of unlabeled groups 
    groups = [batch_size // (nu + 1)] * (nu + 1)
    for x in range(batch_size - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    return offsets

def interleave(mix_group, batch_size):
    '''
    mix the elements of batches
    this code is adopted from: https://github.com/TorchSSL/TorchSSL/tree/main/models/remixmatch
    '''
    
    nu = len(mix_group) - 1
    offsets = interleave_offsets(batch_size, nu)

    # split each group using the same offsets
    mix_group = [[v[offsets[i]:offsets[i+1]] for i in range(nu + 1)] for v in mix_group]

    # swap stripes so each mixed batch has a slice from every group
    for i in range(1, nu + 1):
        mix_group[0][i], mix_group[i][i] = mix_group[i][i], mix_group[0][i]

    # stitch back each mixed batch
    return [torch.cat(v, dim=0) for v in mix_group]

def mixup_one(mixed_inputs, mixed_labels, alpha):
    '''
    MixUp technique.
    code adeopted from: https://github.com/TorchSSL/TorchSSL/tree/main/models/remixmatch
    '''
    lam = torch.distributions.Beta(alpha, alpha).sample().to(mixed_inputs.device)
    idx = torch.randperm(mixed_inputs.size(0), device=mixed_inputs.device)

    mixed_x  = lam * mixed_inputs + (1 - lam) * mixed_inputs[idx]
    mixed_y  = lam * mixed_labels + (1 - lam) * mixed_labels[idx]
    return mixed_x, mixed_y, lam


def create_rotation_data(images):
    '''
    create roation data for the rotation loss using
    the first strongly augmented image
    '''        
    batch_size = images.size(0)

    # create rotation from [0, 90, 180, 270] degress
    rotations = [
        images,                                 
        torch.rot90(images, k=1, dims=[2, 3]),    
        torch.rot90(images, k=2, dims=[2, 3]),   
        torch.rot90(images, k=3, dims=[2, 3])    
    ]

    # image batch [4B, C, H, W]
    rotation_images = torch.cat(rotations, dim=0)         

    # label [4B]
    rotation_labels = torch.arange(4, device=images.device).repeat_interleave(batch_size)  
    return rotation_images, rotation_labels


# function to calculate rotation loss
def rotation_loss(model, rot_head, u_strong_1):

    # store feature learned my early layers of the model
    features = {}

    # this function takes the input features before the fully connceted layer
    def fc_input(module, inputs):
        features['feats'] = inputs[0] 

    # run before fully connceted layer to get the features
    hook = model.fc.register_forward_pre_hook(fc_input)
    
    # getting rotation data
    rotation_image, rotation_label = create_rotation_data(u_strong_1)

    # forward pass
    logits = model(rotation_image)                                 
    rot_logits = rot_head(features['feats']) 
    
    # loss             
    loss_rot = F.cross_entropy(rot_logits, rotation_label)
    hook.remove()
    return loss_rot
    

# finction to calculate cross entropy loss on soft labels
def soft_cross_entropy_loss(logits, soft_label):
    # adopted from: https://github.com/TorchSSL/TorchSSL/tree/main/models/remixmatch
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_label * log_probs).sum(dim=1).mean()


def ramp_up(global_step, warm_up, total_iter=1024*256):
    '''
    ramp up for the pre-mixup and rotation loss weight 
    adopted from: https://github.com/TorchSSL/TorchSSL/tree/main/models/remixmatch
    '''
    return float(np.clip(global_step / (warm_up * total_iter), 0.0, 1.0))


class MovingAverage:
    '''
    running average of model's prediction on unlabeled data
    adopted from: https://github.com/TorchSSL/TorchSSL/tree/main/models/remixmatch
    '''
    def __init__(self, num_class, momentum=0.999):
        self.p_model = (torch.ones(num_class) / num_class).to(device)
        self.momentum = momentum

    @torch.no_grad()
    def update(self, p_logits):
        # average over batch
        p_batch = p_logits.mean(0).to(self.p_model.device)  
        self.p_model = self.momentum * self.p_model + (1 - self.momentum) * p_batch
        # re-normalize
        self.p_model = self.p_model/self.p_model.sum()  


def estimate_prior_distribution(data_path, num_classes=10, device=device):
    # .pt file with (images, labels)
    image, label = torch.load(data_path)
    # label = label.view(-1).long()
    
    # count frequency of each class
    counts = torch.bincount(label, minlength=num_classes).float()

    # prior distribution
    prior_dist = counts / counts.sum().clamp_min(1e-12)
    return prior_dist.to(device)


def cycle_loader(loader):
    '''
    for indifinite data loading
    '''
    while True:
        for batch in loader:
            yield batch

