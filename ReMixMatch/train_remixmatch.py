import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import math
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from remixmatch_dataset import get_train_dataloader_labeled, get_train_dataloader_unlabeled, get_val_dataloader_labeled
from utils import mixup_one, interleave, soft_cross_entropy_loss, ramp_up, estimate_prior_distribution, rotation_loss, MovingAverage
from models.wideresnet import WideResNet 


# GPU setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# get the backbone model
def get_model(model_name, num_classes):
    if model_name == 'wrn':
        model = WideResNet(num_classes=num_classes)
    elif model_name == '':
        pass
    else:
        print('Model not found')
    return model


@torch.no_grad()
def build_pseudo_label(logits_list, T, p_data, p_model_ma, use_dm = True):
   
    '''
    here, this function create pseudo labels for the k strongly augmented image
    for this code we have used:
    1. https://github.com/google-research/remixmatch/tree/master
    2. https://github.com/TorchSSL/TorchSSL/tree/main/models/remixmatch
    '''

    # converting logits to probability
    p_model_list = [F.softmax(l, dim=1) for l in logits_list]  

    # concatinating probabilities
    p_model_distribution = torch.cat(p_model_list, dim=0)          

    if use_dm:
        # distribution alignment
        p_ratio = (p_data.to(p_model_distribution.device) + 1e-6) / (p_model_ma.p_model.to(p_model_distribution.device) + 1e-6)  # [C]
        p_weighted = p_model_distribution * p_ratio.unsqueeze(0)
        # normalize
        p_weighted = p_weighted / p_weighted.sum(dim=1, keepdim=True)
    else:
        p_weighted = p_model_distribution.detach()

    # sharpening
    pseudo_label = p_weighted.pow(1.0 / T)
    pseudo_label = pseudo_label / pseudo_label.sum(dim=1, keepdim=True)

    # here shape of pseudo_label is [B*(K+1), C]
    return pseudo_label.detach(), p_model_distribution.detach()



@torch.no_grad()
def update_ema_model(model, ema_model, ema_decay):
    # update ema model parameters
    for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
        if model_param.requires_grad:
            ema_param.data = (1.0 - ema_decay) * model_param.data + ema_decay * ema_param.data

     # sync batch normalization
    for ema_bn_parameter, model_bn_parameter in zip(ema_model.buffers(), model.buffers()):
        ema_bn_parameter.copy_(model_bn_parameter)


# training function
def training_step(model, ema_model, train_data_loader_labeled, train_data_loader_unlabeled, optimizer, k, T, 
                  p_data, p_model, num_classes, alpha, lambda_u, lambda_u1, lambda_r, warm_up, ema_decay, rotation_head, total_iter, use_dm=True):
    
    model.train()
    ema_model.train()

    # initializing total losses
    total_loss = 0.0
    total_loss_com = 0.0
    total_loss_l = 0.0
    total_loss_u = 0.0

    num_step = 0
    base_lambda_u = lambda_u
    base_lambda_u1 = lambda_u1

    # to calucalte total number of the trainig step and ramp up
    global_step = getattr(training_step, '_global_step', 0)

    # progeress bar setup
    progress_bar = tqdm(zip(train_data_loader_labeled, train_data_loader_unlabeled), total=len(train_data_loader_labeled), desc="Training")

    for index, batch in enumerate(progress_bar):
        # loss = 0.0
        label_batch, unlabel_batch = batch 

        # label data with weak augmentation
        x_l, y_l = label_batch

        # unlabel data with weak augmentation and K strong augmentation shape ([B, C, H, W],  [B, K, C, H, W])
        x_u_weak, x_u_strong_k = unlabel_batch 
        batch_size = x_l.size(0)

        # sending tensor to GPU
        x_l = x_l.to(device)
        y_l = y_l.to(device)
        x_u_weak = x_u_weak.to(device)
        x_u_strong_k = x_u_strong_k.to(device)
        

        # augmentation anchoring
        # forward pass
        logit_x_u_weak = model(x_u_weak) 

        # creating a logit list of len (k+1) to generate pseudo labels as we have k strongly augmented image and one weakly augmented image
        logit_x_u_weak_list = [logit_x_u_weak] * (k+1)

        # getting pseudo labels with DA and sharpening 
        with torch.no_grad():
            y_pseudo, p_model_logits = build_pseudo_label(logit_x_u_weak_list, T, p_data, p_model, use_dm)

            # updating moving average 
            p_model.update(p_model_logits)

        # mixup all labeled and unlabeled data
        # one hot encoding for the lablels on the labeled data
        y_l_onehot = F.one_hot(y_l, num_classes).float()

        # concat all unlabeled images shape of x_u [B*(K+1), C, H, W]
        x_u = torch.cat([x_u_weak] + [x_u_strong_k[:, i] for i in range(k)], dim=0)  
        u_pseudo_l = y_pseudo 
        
        # Mixup
        mixed_inputs = torch.cat([x_l, x_u], dim=0)

        # re-shape psedo labels to (K+1, B, C) to concatenate with one hot y_l_onehot
        u_labels = u_pseudo_l.view(k+1, batch_size, num_classes)   
        mixed_labels = torch.cat([y_l_onehot] + [u_labels[i] for i in range(k+1)], dim=0) # [B + B*(K+1), C] 

        mixed_x, mixed_y, _ = mixup_one(mixed_inputs, mixed_labels, alpha)

        # split from mix
        x_l_mix      = mixed_x[:batch_size] 
        y_l_mix      = mixed_y[:batch_size]
        u_mix        = mixed_x[batch_size:]    
        u_pseudo_mix   = mixed_y[batch_size:] 

        # split unlabeled data into k+1 groups of size B
        u_groups = list(torch.split(u_mix, batch_size, dim=0))   

        # interleave inputs for stable batch normalization
        grouped_batches = interleave([x_l_mix] + u_groups, batch_size=batch_size)

        # forward pass
        logits_batches = [model(b) for b in grouped_batches] 
        logits_u1 = model(x_u_strong_k[:, 0])     
        
        # de-interleave logits back to original group order
        logits = interleave(logits_batches, batch_size=batch_size)

        # labeled logits
        logits_x = logits[0] 
        
        # unlabeled logits                      
        logits_u = torch.cat(logits[1:], dim=0)   

        # calculate loss
        # labeled CE with mixed soft targets
        loss_l = soft_cross_entropy_loss(logits_x, y_l_mix)

        # unlabeled CE with pseudo targets after mixup
        loss_u = soft_cross_entropy_loss(logits_u, u_pseudo_mix)

        # loss U1
        loss_u1 = soft_cross_entropy_loss(logits_u1, u_labels[0])

        # rotation loss
        loss_rot = rotation_loss(model, rotation_head, x_u_strong_k[:, 0])

        
        # combine loss (supervise + unsupervise + rotation loss)
        # ramp up
        lambda_u = ramp_up(global_step, warm_up, total_iter) * base_lambda_u
        lambda_u1 = ramp_up(global_step, warm_up, total_iter)  * base_lambda_u1
        
        loss = loss_l + lambda_u * loss_u + lambda_u1 * loss_u1 + lambda_r * loss_rot

        # zero the parameter gradients 
        optimizer.zero_grad()

        # backward pass
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # update EMA
        update_ema_model(model, ema_model, ema_decay)

        # calculate loss to comapre with validation loss
        with torch.no_grad():
            ema_model.eval()
            logits_label = ema_model(x_l)
            loss_com = F.cross_entropy(logits_label, y_l, reduction='mean')
        
        # update metrics
        # total loss after all the training steps
        total_loss = total_loss + loss.item()
        total_loss_com = total_loss_com + loss_com.item()
        total_loss_l = total_loss_l + loss_l.item()
        total_loss_u = total_loss_u + loss_u.item()

        # number of iteration
        num_step = num_step + 1
        global_step = global_step + 1

        # update tqdm
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "L_sup": f"{loss_l.item():.4f}",
            "L_unsup": f"{loss_u.item():.4f}",
            # "LR": f"{scheduler.get_last_lr()[0]:.4f}"
        })
    
    avg_total_loss = total_loss / num_step
    avg_total_loss_com = total_loss_com / num_step
    avg_loss_l = total_loss_l / num_step
    avg_loss_u = total_loss_u / num_step

    # global_step
    training_step._global_step = global_step

    print(f"\nEpoch Summary:")
    print(f"  Avg Total Loss: {avg_total_loss:.4f}")
    print(f"  Avg Total Loss Com: {avg_total_loss_com:.4f}")
    print(f"  Avg Supervised Loss: {avg_loss_l:.4f}")
    print(f"  Avg Unsupervised Loss: {avg_loss_u:.4f}")

    return avg_total_loss, avg_total_loss_com


# validation step
def val_step(model, data_loader, criterion, val_acc):
    model.eval()
    val_loss = 0.0
    val_acc.reset()

    with torch.no_grad():
        for data in tqdm(data_loader):
            # get inputs
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # metric
            val_loss = val_loss + loss.item()
            val_acc.update(outputs, labels)

    agv_val_loss = val_loss / len(data_loader)
    val_accuracy = val_acc.compute().item()
    return agv_val_loss, val_accuracy


# final test
def test_step(args):
    print('Final Tesing:')
    model = get_model(args.model_name, args.num_class)
    model.load_state_dict(torch.load(args.model_weight))
    model.to(device)
    model.eval()

    eval_loss = 0.0
    test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=args.num_class).to(device)
    test_acc.reset()
    all_label = []
    all_pred = []

    # getting the test data
    test_dataloader = get_val_dataloader_labeled(data_path=args.test_path, batch_size=args.batch_size)

    with torch.no_grad():
        for data in tqdm(test_dataloader):
            # get inputs
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(images)
            criterion = nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(outputs, labels)

            # converting logits to class predictions
            _, predicted = torch.max(outputs, 1)
            
            all_label.extend(labels.to('cpu').numpy())
            all_pred.extend(predicted.to('cpu').numpy())

            # metric
            eval_loss = eval_loss + loss.item()
            test_acc.update(outputs, labels)
            
    eval_loss = eval_loss / len(test_dataloader)
    test_accuracy = test_acc.compute().item()
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print()
    # Classification Report
    print('Classification Report: ')
    print(classification_report(all_label, all_pred))


# training loop
def model_training(args):
    # setting up seed value 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # number of data in training steps
    num_sample = args.train_step*args.batch_size
    total_iter = args.train_step* args.epochs

    # getting data
    train_label_dataloader = get_train_dataloader_labeled(data_path=args.train_path_label, batch_size=args.batch_size, total_image=num_sample)
    train_unlabel_dataloader = get_train_dataloader_unlabeled(data_path=args.train_path_unlabel, batch_size=args.batch_size, k=args.k, total_image=num_sample)
    val_dataloader = get_val_dataloader_labeled(data_path=args.val_path, batch_size=args.batch_size)

    # prior distribution
    p_data = estimate_prior_distribution(args.train_path_label, args.num_class, device=device)

    # getting model 
    model = get_model(args.model_name, num_classes=args.num_class).to(device)
    ema_model = get_model(args.model_name, num_classes=args.num_class).to(device)
    ema_model.load_state_dict(model.state_dict())

    # frezee gradient update for EMA model
    for param in ema_model.parameters():
        param.requires_grad_(False)

    # rotation head (for rotation loss)
    feat_dim = model.fc.in_features 
    rotation_head = nn.Linear(feat_dim, 4).to(next(model.parameters()).device)

    # moving average for DA
    p_model = MovingAverage(args.num_class)
    
    # setting up loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # optimizer
    # optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    # adding rotaion head params to optimizer
    optimizer.add_param_group({'params': rotation_head.parameters()})
    
    # define metrics
    val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=args.num_class).to(device)

    best_val_accuracy = 0.0

    # list for saving metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}:')
        # training step
        total_training_loss, train_loss = training_step(model, ema_model, train_label_dataloader, train_unlabel_dataloader, optimizer, args.k, args.T,
                                    p_data, p_model, args.num_class, args.alpha, args.lambda_u, args.lambda_u1, args.lambda_r, 
                                    args.warm_up, args.ema_decay, rotation_head, total_iter, args.use_dm)
        
        # validation step
        val_loss, val_accuracy = val_step(ema_model, val_dataloader, criterion, val_acc)


        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}')

        # storing metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
    
        # save model
        os.makedirs(args.save_dir, exist_ok=True)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = os.path.join(args.save_dir, f'ReMixMatch_{args.model_name}_best_model_10_pct_k4.pt')
            torch.save(ema_model.state_dict(), checkpoint_path)
            print(f"Model saved with Validation Accuracy: {best_val_accuracy:.4f}")

    print('Training Finished') 

    # ploting train vs val loss 
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss') 
    plt.legend()
    plt.savefig('./plots/remixmatch_loss_curve_10_pct_k2_without_u1_n_rot', dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--train-path-label', type=str, required=True, help='train split path label')
    parser.add_argument('--train-path-unlabel', type=str, required=True, help='train split path unlabel')
    parser.add_argument('--val-path', type=str, required=True, help='val split path')
    parser.add_argument('--test-path', type=str, required=True, help='test split path')
    parser.add_argument('--num-class', type=int, default=10)
    parser.add_argument('--model-name', type=str, default='wrn', help='Model name')
    parser.add_argument('--epochs', type=int, default=1024)
    parser.add_argument('--train-step', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--lambda_u', type=float, default=1.5)
    parser.add_argument('--lambda_u1', type=float, default=0.5)
    parser.add_argument('--lambda_r', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.75)
    parser.add_argument('--weight-decay', type=float, default=0.02)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--warm-up', type=float, default=1 / 64)
    parser.add_argument('--use-dm', type=bool, default=True)
    parser.add_argument('--save-dir', type=str, default='./saved_model')
    parser.add_argument('--model-weight', type=str, default='./saved_model/wrn_best_model.pt')
    args = parser.parse_args()

    model_training(args)
    test_step(args)
