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
from fixmatch_dataset import get_train_dataloader_labeled, get_train_dataloader_unlabeled, get_val_dataloader_labeled
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


# learning rate scheduler (cosine learning rate decay)
def get_scheduler(total_step, optimizer):
    def cosine_lr_decay(k, total_steps):
        return max(0.0, math.cos(math.pi * 7 * k / (16 * total_steps)))
    
    return optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: cosine_lr_decay(x, total_step)
    )


def get_wd_param_list(model):
    '''
    Filter model parameters into two groups. Parameters which should use weight decay and
    parameters which do not
    from https://github.com/fbuchert/fixmatch-pytorch
    '''
    wd_params, no_wd_params = [], []
    for name, param in model.named_parameters():
        # filter batch nomalization parameters from weight decay parameters
        if 'bn' in name:
            no_wd_params.append(param)
        else:
            wd_params.append(param)
    return [{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}]


# training step for a single epoch
def training_step(model, train_data_loader_labeled, train_data_loader_unlabeled, optimizer, criterion, scheduler, threshold=0.95, lambda_u=1):
    
    model.train()
    # initializing total losses
    total_loss = 0.0
    total_loss_l = 0.0
    total_loss_u = 0.0
    num_step = 0

    # progeress bar setup
    progress_bar = tqdm(zip(train_data_loader_labeled, train_data_loader_unlabeled), total=len(train_data_loader_labeled), desc="Training")

    for index, batch in enumerate(progress_bar):
        loss = 0.0

        label_batch, unlabel_batch = batch

        # label data with weak augmentation
        x_l, y_l = label_batch

        # unlabel data with weak augmentation and strong augmentation
        x_u_weak, x_u_strong = unlabel_batch

        # concatenating label and unlabel data
        x_in = torch.cat((x_l, x_u_weak, x_u_strong)).to(device) 
        y_l = y_l.to(device)

        # forward pass
        logits = model(x_in)

        # spliting the model ouputs for label and unlabeled data
        logit_x_l = logits[:len(x_l)]
        logit_x_u_weak, logit_x_u_strong = logits[len(x_l):].chunk(2)
        del x_in

        # supervise loss
        loss_l = criterion(logit_x_l, y_l)

        # generating pseudo labels
        with torch.no_grad():
            probs = torch.softmax(logit_x_u_weak, dim=1)
            max_probs, y_pseudo = torch.max(probs, dim=1)
            # creating mask to calculate unsupervise loss (0 where probability is less than threshold)
            mask = max_probs.ge(threshold).float()
        
        loss_u = (F.cross_entropy(logit_x_u_strong, y_pseudo, reduction='none') * mask).mean()

        # combine loss (supervise + unsupervise)
        loss = loss_l.mean() + lambda_u * loss_u
        
        # zero the gradients 
        optimizer.zero_grad()

        # backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # update metrics
        # total loss after all the training steps
        total_loss = total_loss + loss.item()
        total_loss_l = total_loss_l + loss_l.item()
        total_loss_u = total_loss_u + loss_u.item()
        

        # number of iteration
        num_step = num_step + 1

        # update tqdm
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'L_sup': f'{loss_l.item():.4f}',
            'L_unsup': f'{loss_u.item():.4f}',
            'LR': f'{scheduler.get_last_lr()[0]:.4f}'
        })
    
    # compute average losses
    avg_total_loss = total_loss / num_step
    avg_loss_l = total_loss_l / num_step
    avg_loss_u = total_loss_u / num_step

    print(f'\nEpoch Summary:')
    print(f'  Avg Total Loss: {avg_total_loss:.4f}')
    print(f'  Avg Supervised Loss: {avg_loss_l:.4f}')
    print(f'  Avg Unsupervised Loss: {avg_loss_u:.4f}')

    return avg_total_loss


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
    # getting model
    model = get_model(args.model_name, args.num_class)
    # load model weights 
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
            _ , predicted = torch.max(outputs, 1)
            
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

    # number of labeled sample in training steps
    num_samples_label = args.train_step*args.batch_size

    # getting data
    train_label_dataloader = get_train_dataloader_labeled(data_path=args.train_path_label, batch_size=args.batch_size, num_samples_label=num_samples_label)
    train_unlabel_dataloader = get_train_dataloader_unlabeled(data_path=args.train_path_unlabel, batch_size=args.batch_size, num_samples_label=num_samples_label, mu=args.mu)
    val_dataloader = get_val_dataloader_labeled(data_path=args.val_path, batch_size=args.batch_size)

    # getting model 
    model = get_model(args.model_name, args.num_class).to(device)

    # Setting up loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # optimizer
    model_parameters_optim = get_wd_param_list(model)
    optimizer = optim.SGD(model_parameters_optim, lr=args.learning_rate, momentum=args.beta, weight_decay=args.weight_decay, nesterov=True)

    # learning rate scheduler
    scheduler = get_scheduler(args.train_step * args.epochs, optimizer)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_step * args.epochs)

    # defined metrics
    val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=args.num_class).to(device)
    best_val_accuracy = 0.0

    # list for saving metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}:')
        # training step
        train_loss = training_step(model, train_label_dataloader, train_unlabel_dataloader, optimizer,  criterion, scheduler, args.threshold, args.lambda_u)
        # validation step
        val_loss, val_accuracy = val_step(model, val_dataloader, criterion, val_acc)

        print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        #  storing metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # save model
        os.makedirs(args.save_dir, exist_ok=True)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = os.path.join(args.save_dir, f'fixmatch_{args.model_name}_best_model_10_pct_cifar10.pt')
            torch.save(model.state_dict(), checkpoint_path)
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
    plt.savefig('./plots/fixmatch_loss_curve_10_pct_cifar10', dpi=300)
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
    parser.add_argument('--learning-rate', type=float, default=0.03)
    parser.add_argument('--mu', type=int, default=7)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--lambda_u', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--save-dir', type=str, default='./saved_model')
    parser.add_argument('--model-weight', type=str, default='./saved_model/fixmatch_wrn_best_model.pt')
    args = parser.parse_args()

    # training loop
    model_training(args)
    # final test (on independent test dataset)
    test_step(args)