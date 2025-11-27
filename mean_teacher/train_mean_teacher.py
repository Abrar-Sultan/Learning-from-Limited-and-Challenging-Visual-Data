import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import math
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from mean_teacher_dataset import get_train_dataloader_labeled, get_train_dataloader_unlabeled, get_val_dataloader_labeled, NO_LABEL
from models.wideresnet import WideResNet 
from utils import sigmoid_rampup, learning_rate_scheduler, softmax_mse_loss


# to calucalte total number of the trainig step and for EMA update
global_step = 0

# GPU setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


# get the backbone model
def get_model(model_name):
    if model_name == 'wrn':
        model = WideResNet()
    elif model_name == '':
        pass
    else:
        print('Model not found')
    return model


# update the teacher model with exponential moving average (EMA)
@torch.no_grad()
def update_teacher_model(student_model, teacher_model, ema_decay, global_step):

    # compute dynamic alpha for early steps
    alpha = min(1.0 - 1.0 / (global_step + 1), ema_decay)

    # update teacher model parameters
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        # ema = alpha * teacher_param + (1 - alpha) * student_param
        teacher_param.mul_(alpha).add_(student_param, alpha=(1.0 - alpha))


# training step for a single epoch
def training_step(args, student_model, teacher_model, train_data_loader_labeled, train_data_loader_unlabeled, 
                  optimizer, criterion, consistency_criterion, epoch, consistency=100, consistency_rampup=5, ema_decay=0.999): 
    
    global global_step
    student_model.train()
    teacher_model.train()

    # initializing total losses
    total_loss = 0.0
    total_loss_l = 0.0
    total_loss_u = 0.0
    total_cosistancy_loss = 0.0
    num_step = 0

    # initializing accuracy metrics
    train_acc_student = torchmetrics.Accuracy(task='multiclass', num_classes=args.num_class).to(device)
    train_acc_teacher = torchmetrics.Accuracy(task='multiclass', num_classes=args.num_class).to(device)

    # progeress bar setup
    progress_bar = tqdm(zip(train_data_loader_labeled, train_data_loader_unlabeled), total=len(train_data_loader_labeled), desc='Training')

    for index, (label_batch, unlabel_batch) in enumerate(progress_bar):
        # label data with two augmentation
        (x_l1, x_l2), y_l = label_batch
        # unlabel data with two augmentation
        (x_u1, x_u2), y_u = unlabel_batch # y_u is -1

        # combine the batches
        input_x_student = torch.cat([x_l1, x_u1], dim=0).to(device)
        input_x_teacher = torch.cat([x_l2, x_u2], dim=0).to(device)
        label_y = torch.cat([y_l, y_u], dim=0).to(device)

        # creating mask to calculate training accuracy for the label data
        mask = (label_y != NO_LABEL)

        # batch infomation 
        batch_size = label_y.size(0)

        # forward pass
        with torch.no_grad():
            teacher_model_logits = teacher_model(input_x_teacher)

        student_model_logits = student_model(input_x_student)

        # detach as teacher model should not get gradients
        teacher_model_logits = teacher_model_logits.detach()

        # classification loss
        loss_l = criterion(student_model_logits, label_y) / batch_size

        # teacher model classification loss
        loss_u = criterion(teacher_model_logits, label_y) / batch_size

        # consistancy loss
        consistancy_weight = consistency * sigmoid_rampup(epoch, consistency_rampup)
        consistancy_loss = consistancy_weight * consistency_criterion(student_model_logits, teacher_model_logits) / batch_size
        
        # total loss
        loss = loss_l + consistancy_loss

        # zero the parameter gradients 
        optimizer.zero_grad()

        # backward pass
        loss.backward()
        optimizer.step()

        # learnng rate scheduler update
        lr = learning_rate_scheduler(optimizer, epoch, index, len(train_data_loader_labeled), args)

        # updating teacher models parameters
        global_step = global_step + 1
        num_step = num_step + 1
        update_teacher_model(student_model, teacher_model, ema_decay, global_step)

        # update metrics
        # total loss after all the training steps
        total_loss = total_loss + loss.item()
        total_cosistancy_loss = total_cosistancy_loss + consistancy_loss.item()
        total_loss_l = total_loss_l + loss_l.item()
        total_loss_u = total_loss_u + loss_u.item()

        # Update accuracy for labeled samples only
        if mask.any():
            train_acc_student.update(student_model_logits[mask], label_y[mask])
            train_acc_teacher.update(teacher_model_logits[mask], label_y[mask])

        # update tqdm
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cosistancy_loss': f'{loss.item():.4f}',
            'L_student': f'{loss_l.item():.4f}',
            'L_teacher': f'{loss_u.item():.4f}',
            'lr': f'{lr:.4f}'
        })
    
    # compute average losses
    avg_total_loss = total_loss / num_step
    avg_consistancy_loss = total_cosistancy_loss / num_step
    avg_loss_l = total_loss_l / num_step
    avg_loss_u = total_loss_u / num_step

    # compute accuracies
    train_accuracy_student = train_acc_student.compute().item()
    train_accuracy_teacher = train_acc_teacher.compute().item()
    

    print(f"\nEpoch Summary:")
    print(f"  Training Loss: {avg_total_loss:.4f}")
    print(f"  Avg Consistancy Loss: {avg_consistancy_loss:.4f}")
    print(f"  Avg Student Loss: {avg_loss_l:.4f}")
    print(f"  Avg Teacher Loss: {avg_loss_u:.4f}")
    print(f"  Student Train Accuracy: {train_accuracy_student:.2f}")
    print(f"  Teacher Train Accuracy: {train_accuracy_teacher:.2f}")
    
    return avg_total_loss, train_accuracy_student, train_accuracy_teacher
        

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

            batch_size = labels.size(0)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels) / batch_size

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
    model = get_model(args.model_name)
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

    global global_step

    # sample in training steps
    label_batch_size = args.batch_size // args.label_unlabel_ratio
    unlabel_batch_size = args.batch_size - label_batch_size

    num_samples_label = args.train_step*label_batch_size
    num_samples_unlabel = args.train_step * unlabel_batch_size

    # getting data
    train_label_dataloader = get_train_dataloader_labeled(data_path=args.train_path_label, batch_size=label_batch_size, num_samples_label=num_samples_label)
    train_unlabel_dataloader = get_train_dataloader_unlabeled(data_path=args.train_path_unlabel, batch_size=unlabel_batch_size, num_samples_unlabel=num_samples_unlabel)
    val_dataloader = get_val_dataloader_labeled(data_path=args.val_path, batch_size=args.batch_size)

    # getting model 
    student_model = get_model(args.model_name).to(device)
    teacher_model = get_model(args.model_name).to(device)

    for param in teacher_model.parameters():
        param.requires_grad_(False)
    

    # Setting up loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL)
    consistency_criterion = softmax_mse_loss

    # optimizer
    optimizer = optim.SGD(student_model.parameters(), lr=args.learning_rate, momentum=args.beta, weight_decay=args.weight_decay, nesterov=True)
    

    # define metrics
    val_acc_student = torchmetrics.Accuracy(task='multiclass', num_classes=args.num_class).to(device)
    val_acc_teacher = torchmetrics.Accuracy(task='multiclass', num_classes=args.num_class).to(device)

    best_val_accuracy = 0.0

    # list for saving metrics
    train_losses = []
    val_losses_student = []
    val_accuracies_student = []
    val_losses_teacher = []
    val_accuracies_teacher = []
    train_accuracies_student = []
    train_accuracies_teacher = []

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}:')
        # training step
        train_loss, train_accuracy_student, train_accuracy_teacher = training_step(args, student_model, teacher_model, train_label_dataloader, train_unlabel_dataloader, optimizer, criterion, consistency_criterion, epoch, args.consistency, args.consistency_rampup, args.ema_decay)
        # validation step
        val_loss_student, val_accuracy_student = val_step(student_model, val_dataloader, criterion, val_acc_student)
        val_loss_teacher, val_accuracy_teacher = val_step(teacher_model, val_dataloader, criterion, val_acc_teacher)

        print(f'Validation Loss: {val_loss_student:.4f}, Student Validation Accuracy: {val_accuracy_student:.2f}, Teacher Validation Accuracy: {val_accuracy_teacher:.2f}')

        # storing metrics
        train_losses.append(train_loss)
        val_losses_student.append(val_loss_student)
        val_losses_teacher.append(val_loss_teacher)

        train_accuracies_student.append(train_accuracy_student)
        train_accuracies_teacher.append(train_accuracy_teacher)
        val_accuracies_student.append(val_accuracy_student)
        val_accuracies_teacher.append(val_accuracy_teacher)

        # save model
        is_best = val_accuracy_teacher > best_val_accuracy 
        best_val_accuracy = max(val_accuracy_teacher, best_val_accuracy)

        os.makedirs(args.save_dir, exist_ok=True)
        if is_best:
            student_path = os.path.join(args.save_dir, f'mean_teacher_{args.model_name}_student_best_10_pct.pt')
            teacher_path = os.path.join(args.save_dir, f'mean_teacher_{args.model_name}_ema_best_model_10_pct.pt')

            torch.save(student_model.state_dict(), student_path)
            torch.save(teacher_model.state_dict(), teacher_path)

            print(f'Teacher model saved with Validation Accuracy: {best_val_accuracy:.4f}')

    print('Training Finished')

    # ploting train vs val loss and accuracy curve
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses_student, label='Validation Loss Student')
    plt.plot(range(1, args.epochs + 1), val_losses_teacher, label='Validation Loss Teacher')
    plt.xlabel('Epoch') 
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss (Teacher and Student Model)')
    plt.legend()
    plt.savefig('./plots/mean_teacher_loss_curve_10_pct.png', dpi=300) 
    plt.show()

    # plot student model accuracy
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, args.epochs + 1), train_accuracies_student, label='Train Accuracy student')
    plt.plot(range(1, args.epochs + 1), val_accuracies_student, label='Validation Accuracy student')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy (Student Model)')
    plt.legend()
    plt.savefig('./plots/mean_teacher_accuracy_curve_student_1_pct_test.png', dpi=300)
    plt.show()

    # plot teacher model accuracy 
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, args.epochs + 1), train_accuracies_teacher, label='Train Accuracy teacher')
    plt.plot(range(1, args.epochs + 1), val_accuracies_teacher, label='Validation Accuracy teacher')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy (Teacher Model)')
    plt.legend()
    plt.savefig('./plots/mean_teacher_accuracy_curve_teacher_10_pct.png', dpi=300)
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
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--train-step', type=int, default=149)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--label-unlabel-ratio', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--initial-learning-rate', type=float, default=0.0)
    parser.add_argument('--lr-rampup', type=int, default=0)
    parser.add_argument('--lr-rampdown-epochs', type=int, default=210)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--consistency', type=float, default=100.0)
    parser.add_argument('--consistency-rampup', type=int, default=5)    
    parser.add_argument('--save-dir', type=str, default='./saved_model')
    parser.add_argument('--model-weight', type=str, default='./saved_model/fixmatch_wrn_best_model.pt')
    args = parser.parse_args()

    # training loop
    model_training(args)

    # final test
    test_step(args)