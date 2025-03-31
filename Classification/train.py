import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

from data import EmbryoDataset
import helpers


configs = [
    [{'dir': 'data/real/train', 'num_images': 0}, {'dir': 'data/synthetic/GAN', 'num_images': 0}, {'dir': 'data/synthetic/LDM', 'num_images': 500}],
    [{'dir': 'data/real/train', 'num_images': 0}, {'dir': 'data/synthetic/GAN', 'num_images': 500}, {'dir': 'data/synthetic/LDM', 'num_images': 0}],
    [{'dir': 'data/real/train', 'num_images': 0}, {'dir': 'data/synthetic/GAN', 'num_images': 250}, {'dir': 'data/synthetic/LDM', 'num_images': 250}],
    [{'dir': 'data/real/train', 'num_images': 0}, {'dir': 'data/synthetic/GAN', 'num_images': 500}, {'dir': 'data/synthetic/LDM', 'num_images': 500}],    
    [{'dir': 'data/real/train', 'num_images': 0}, {'dir': 'data/synthetic/GAN', 'num_images': 1000}, {'dir': 'data/synthetic/LDM', 'num_images': 1000}],
    # [{'dir': 'data/real/train', 'num_images': 0}, {'dir': 'data/synthetic/GAN', 'num_images': 2000}, {'dir': 'data/synthetic/LDM', 'num_images': 2000}],  
    # [{'dir': 'data/real/train', 'num_images': 0}, {'dir': 'data/synthetic/GAN', 'num_images': 3000}, {'dir': 'data/synthetic/LDM', 'num_images': 3000}],
    # [{'dir': 'data/real/train', 'num_images': 0}, {'dir': 'data/synthetic/GAN', 'num_images': 4000}, {'dir': 'data/synthetic/LDM', 'num_images': 4000}],
    # [{'dir': 'data/real/train', 'num_images': 0}, {'dir': 'data/synthetic/GAN', 'num_images': 5000}, {'dir': 'data/synthetic/LDM', 'num_images': 5000}], 
    [{'dir': 'data/real/train', 'num_images': 1000}, {'dir': 'data/synthetic/GAN', 'num_images': 0}, {'dir': 'data/synthetic/LDM', 'num_images': 0}], 
    # [{'dir': 'data/real/train', 'num_images': 1000}, {'dir': 'data/synthetic/GAN', 'num_images': 500}, {'dir': 'data/synthetic/LDM', 'num_images': 500}],
    # [{'dir': 'data/real/train', 'num_images': 1000}, {'dir': 'data/synthetic/GAN', 'num_images': 1000}, {'dir': 'data/synthetic/LDM', 'num_images': 1000}],
    # [{'dir': 'data/real/train', 'num_images': 1000}, {'dir': 'data/synthetic/GAN', 'num_images': 2000}, {'dir': 'data/synthetic/LDM', 'num_images': 2000}],  
    # [{'dir': 'data/real/train', 'num_images': 1000}, {'dir': 'data/synthetic/GAN', 'num_images': 3000}, {'dir': 'data/synthetic/LDM', 'num_images': 3000}],
    # [{'dir': 'data/real/train', 'num_images': 1000}, {'dir': 'data/synthetic/GAN', 'num_images': 4000}, {'dir': 'data/synthetic/LDM', 'num_images': 4000}],
    # [{'dir': 'data/real/train', 'num_images': 1000}, {'dir': 'data/synthetic/GAN', 'num_images': 5000}, {'dir': 'data/synthetic/LDM', 'num_images': 5000}], 
    ]

MODEL = 'vgg'  # vgg, vit, resnet
PRETRAINED = False
NUM_CLASSES = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

helpers.set_seed(123)

transforms = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

def train():
    train_dataset = EmbryoDataset(DIRS_INFO, transform=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = EmbryoDataset(dirs_info=[{'dir': 'data/real/validation', 'num_images': None}], transform=transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001)

    num_epochs = 200
    max_accuracy = 0  
    early_stop_counter = 0


    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_predictions = []
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())

        val_loss /= len(val_dataset)
        val_accuracy = 100 * np.mean(np.array(all_val_predictions) == np.array(all_val_labels))
        val_precision = precision_score(all_val_labels, all_val_predictions, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_predictions, average='weighted')
        val_f1 = f1_score(all_val_labels, all_val_predictions, average='weighted')
        val_mcc = matthews_corrcoef(all_val_labels, all_val_predictions)

        print(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy}%, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}, MCC: {val_mcc:.4f}")

        if val_accuracy > max_accuracy:
            torch.save(model, f'checkpoints/{MODEL}')
            max_accuracy = val_accuracy
            print("Saved checkpoint")
            early_stop_counter = 0

        early_stop_counter += 1
        if early_stop_counter == 30:
            print("Early stopping triggered.")
            break

        scheduler.step(val_loss)


def test():
    model = torch.load(f'checkpoints/{MODEL}')
    model.eval()

    test_dataset = EmbryoDataset(dirs_info=[{'dir': 'data/real/test', 'num_images': None}], transform=transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * np.mean(np.array(all_predictions) == np.array(all_labels))
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_predictions)

    print(f"TEST -> Accuracy: {accuracy}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}\n")
    return accuracy, precision, recall, f1, mcc


for DIRS_INFO in configs:
    model = helpers.get_model(MODEL, PRETRAINED, NUM_CLASSES).to(device)

    train()
    accuracy, precision, recall, f1, mcc = test()

    # Save checkpoints
    dirs_info_str = "_".join([f"{info['num_images']}" for info in DIRS_INFO])
    checkpoint_filename = f"{MODEL}_{PRETRAINED}_{dirs_info_str}_ckpt"
    torch.save(model.state_dict(), f'checkpoints/{checkpoint_filename}')

    # Write results
    with open(f'results_{MODEL}_{PRETRAINED}.txt', 'a') as file:
        file.write(f"Dirs Info: {DIRS_INFO}, Pretrained: {PRETRAINED}, ")
        file.write(f"Accuracy: {accuracy}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, MCC: {mcc:.4f}\n")
