import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from data_for_supervised import get_dataloader
from models.wideresnet import WideResNet  

# GPU setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# get model
def get_model(model_name, num_classes):
    if model_name == 'wrn':
        model = WideResNet(num_classes=num_classes)
    elif model_name == '':
        pass
    else:
        print('Model not found')
    return model


# training step for a single epoch
def training_step(model, data_loader, optimizer, criterion, train_acc):
    model.train()
    running_loss = 0.0
    train_acc.reset()

    for data in tqdm(data_loader):
        # get inputs
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        # zero the gradients 
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()
        optimizer.step()

        # metric
        running_loss = running_loss + loss.item()
        train_acc.update(outputs, labels)

    # average training loss
    train_loss = running_loss/len(data_loader)

    # training accuracy
    train_accuracy = train_acc.compute().item()
    return train_loss, train_accuracy


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

    avg_val_loss = val_loss / len(data_loader)
    val_accuracy = val_acc.compute().item()
    return avg_val_loss, val_accuracy


# final test
def test_step(args):
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
    test_dataloader = get_dataloader(data_path=args.test_path, batch_size=args.batch_size, train=False)

    with torch.no_grad():
        for data in tqdm(test_dataloader):
            # get inputs
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(images)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)

            # converting logits to class predictions 
            _ , predicted = torch.max(outputs, 1)
            
            all_label.extend(labels.to('cpu').numpy())
            all_pred.extend(predicted.to('cpu').numpy())

            # update metrics
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

    # getting data
    train_dataloader = get_dataloader(data_path=args.train_path, batch_size=args.batch_size)
    val_dataloader = get_dataloader(data_path=args.val_path, batch_size=args.batch_size, train=False)

    # getting model 
    model = get_model(args.model_name, args.num_class).to(device)

    # Setting up loss fucntion and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)


    # defined metrics
    train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=args.num_class).to(device)
    val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=args.num_class).to(device)
    best_val_accuracy = 0.0

    # list for saving metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}:')
        # training step
        train_loss, train_accuracy = training_step(model, train_dataloader, optimizer, criterion, train_acc)
        # validation step
        val_loss, val_accuracy = val_step(model, val_dataloader, criterion, val_acc)

        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}')

        # storing metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # saving model
        os.makedirs(args.save_dir, exist_ok=True)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint_path = os.path.join(args.save_dir, f'{args.model_name}_best_model_cifar10.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved with Validation Accuracy: {best_val_accuracy:.2f}") 

    print('Training Finished')

   # ploting train vs val loss and accuracy curve
    plt.figure(figsize=(5, 5))

    # ploting loss
    # plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch') 
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.savefig('./plots/supervise_loss_curve_cifar10.png', dpi=300)
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(5, 5))
    plt.plot(range(1, args.epochs + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, args.epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.savefig('./plots/supervise_accuracy_curve_cifar10.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--train-path', type=str, required=True, help='train split path')
    parser.add_argument('--val-path', type=str, required=True, help='val split path')
    parser.add_argument('--test-path', type=str, required=True, help='test split path')
    parser.add_argument('--num-class', type=int, default=10)
    parser.add_argument('--model-name', type=str, default='wrn', help='Model name')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--save-dir', type=str, default='./saved_model')
    parser.add_argument('--model-weight', type=str, default='./saved_model/wrn_best_model.pt')
    args = parser.parse_args()

    model_training(args)
    test_step(args) 


