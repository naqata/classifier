from animals_dataset import AnimalDataset
from animals_model import AnimalModel

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Animal Clasifier Training Script')
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--image-size', '-i', type=int, default=224, help='Image size')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume training')
    parser.add_argument('--data-path', '-d', type=str, default='../animals', help='Dataset path')
    parser.add_argument('--tensorboard-dir', '-t', type=str, default='./logs', help='Tensorboard directory')
    parser.add_argument('--checkpoint-dir', '-c', type=str, default='./checkpoints', help='Checkpoint directory')

    return parser.parse_args()

# copy on internet
def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    #color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap='Purples')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > threshold else 'black'
            plt.text(j, i, cm[i, j], horizontalalignment='center', color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('Confusion_matrix', figure, epoch)


def train(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure directories exist
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data transformations
    transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor()
    ])

    # Datasets and Loaders
    train_set = AnimalDataset(root=args.data_path, train=True, transform=transform)
    val_set = AnimalDataset(root=args.data_path, train=False, transform=transform)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=(device.type=='cuda'))

    # Model, Loss, Optimizer
    # model = AnimalModel(num_classes=len(train_set.classes)).to(device)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = nn.Linear(in_features=model.fc.in_features, out_features=len(train_set.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # TensorBoard writer
    writer = SummaryWriter(args.tensorboard_dir)

    # Load checkpoint if resuming
    best_val_loss = np.inf
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'last_model.pt'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        train_progress_bar = tqdm(train_loader, colour='magenta')

        for iter, (images, labels) in enumerate(train_progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            train_progress_bar.set_description(f'Epoch:{epoch+1}/{args.epochs}, Loss:{loss.item():.4f}')

        train_loss /= len(train_loader)
        writer.add_scalar(tag='Train/Loss', scalar_value=train_loss, global_step=epoch)

        # Validation
        model.eval()
        val_loss = 0
        all_labels = []
        all_predictions = []
        val_progress_bar = tqdm(val_loader, colour='green', desc='Validatin:')
        with torch.no_grad():
            for images, labels in val_progress_bar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.tolist())
                all_labels.extend(labels.tolist())
        
        # Calculate metrics and log to TensorBoard
        val_loss /= len(val_loader)
        val_acc = accuracy_score(y_true=all_labels, y_pred=all_predictions)

        writer.add_scalar(tag='Val/loss', scalar_value=val_loss, global_step=epoch)
        writer.add_scalar(tag='Val/Accuracy', scalar_value=val_acc, global_step=epoch)
        plot_confusion_matrix(writer, confusion_matrix(y_true=all_labels, y_pred=all_predictions), train_set.classes, epoch)

        # save checkpoint 
        checkpoint = {
            'classess': train_set.classes,
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }

        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'last_model.pt'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f'::Model saved with validation loss:{val_loss:.4f}, accuracy: {(val_acc*100):2f}%\n')
        else:
            print(f'Val_loss:{val_loss:4f}, Val_Accuracy:{(val_acc*100):.2f}%\n')


if __name__ == '__main__':
    args = parse_arguments()
    train(args)