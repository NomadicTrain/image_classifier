import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torchvision.models import resnet18, resnet34, resnet50
from torch.optim.lr_scheduler import StepLR
import multiprocessing
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = resnet18(pretrained=True)
    elif model_name == 'resnet34':
        model = resnet34(pretrained=True)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def main(args):
    setup_logging()
    logging.info(f"Current working directory: {os.getcwd()}")

    try:
        # Create necessary directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data')
        checkpoint_dir = os.path.join(base_dir, 'checkpoints')

        for directory in [data_dir, checkpoint_dir]:
            os.makedirs(directory, exist_ok=True)

        # Check if CUDA is available and set the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Data preprocessing and augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Load model
        model = get_model(args.model, len(classes)).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

        train_losses = []
        test_accuracies = []
        test_f1_scores = []

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{args.epochs}")
            for i, data in pbar:
                inputs, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (i + 1)})
            
            train_losses.append(running_loss / len(trainloader))
            
            # Evaluation
            model.eval()
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            accuracy = 100 * correct / total
            f1 = f1_score(all_labels, all_preds, average='weighted')
            test_accuracies.append(accuracy)
            test_f1_scores.append(f1)
            logging.info(f'Epoch {epoch + 1} Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}')
            
            if epoch % 5 == 4:  # Save every 5 epochs
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            scheduler.step()

        logging.info('Finished Training')

        # Final evaluation
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        final_accuracy = 100 * correct / total
        final_f1 = f1_score(all_labels, all_preds, average='weighted')
        logging.info(f'Final Accuracy on 10000 test images: {final_accuracy:.2f}%')
        logging.info(f'Final F1 Score: {final_f1:.4f}')

        # Visualization
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.subplot(1, 3, 2)
        plt.plot(test_accuracies)
        plt.title('Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')

        plt.subplot(1, 3, 3)
        plt.plot(test_f1_scores)
        plt.title('Test F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')

        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'training_results.png'))
        plt.close()

        # Save the final model
        torch.save(model.state_dict(), os.path.join(base_dir, 'cifar_resnet_final.pth'))

        logging.info("Training complete. Model saved.")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description='CIFAR-10 Image Classifier')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 20)')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--num-workers', type=int, default=2, help='number of worker processes for data loading (default: 2)')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'], help='model architecture (default: resnet18)')
    args = parser.parse_args()
    main(args)
