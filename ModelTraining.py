import os

import numpy as np
import pandas as pd

from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

import librosa

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

import torchvision
import torchvision.transforms.v2 as v2
from torchvision.io import decode_image
from torchvision.models import resnet34, ResNet34_Weights

from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC

from sklearn.model_selection import train_test_split 

class AudioDataset(Dataset):
    def __init__(self, tsv_path=os.path.join('Utils', 'targets.tsv'), type='train', img_size=(224, 224)):
        self.targets = pd.read_csv(tsv_path, sep='\t', header=None, names=['Name', 'Label'])
        self.size = len(self.targets)
        self.img_size = img_size
        self.type = type
        self.features = pd.read_csv(os.path.join('Utils', f'{type}_features.csv'), index_col=False, header=0)

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        name = self.targets.iloc[index, 0]
        path = os.path.join('Data', f'{self.type}_img', f'{name}.png')
    
        img = decode_image(input=path, mode=torchvision.io.ImageReadMode.RGB)
        transforms = v2.Compose([
            v2.Resize(size=self.img_size), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.1953, 0.0678, 0.2199], std=[0.2656, 0.1031, 0.2017])])
        img = transforms.forward(img)
        
        ser = np.array(self.features[self.features['name'] == name].values[0, 1:], dtype=np.float32)
        data = torch.tensor(ser)

        label = torch.tensor(self.targets.iloc[index, 1], dtype=torch.float32).reshape(1)

        return img, data, label

    def compute_mean_and_std(self):
        
        res_sum: torch.Tensor = None

        for img, label in tqdm(self):
            part_sum = torch.sum(img, dim=[1, 2])

            if res_sum is None:
                res_sum = part_sum
            else:
                res_sum += part_sum

        mean = res_sum / (len(self)*224*224)

        resid_sq_sum = None
        for img, label in tqdm(self):
            for i in range(3):
                cur_mean = mean[i].item()
                img[i] -= cur_mean
            img = img.pow(exponent=2)

            if resid_sq_sum is None:
                resid_sq_sum = torch.sum(img, dim=[1, 2])
            else:
                resid_sq_sum += torch.sum(img, dim=[1, 2])

        resid_sq_sum = resid_sq_sum / (len(self)*224*224-1)
        std = torch.sqrt(resid_sq_sum)

        print(mean)
        print(std)

class CNNModel(nn.Module):
    def __init__(self, tabular_dim=41, out_ft=1):
        super().__init__()
        self.out_ft = out_ft
        self.pretrained = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.fc1 = nn.Linear(self.pretrained.fc.in_features, 100)
        self.fc2 = nn.Linear(100 + tabular_dim, 250)
        self.fc3 = nn.Linear(250, out_ft)

        self.pretrained.fc = nn.Identity()

        self.dropout = nn.Dropout(p=0.4)

    def forward(self, img, data):
        x1 = self.pretrained.forward(img)
        x1 = self.fc1(x1)

        x = torch.cat((x1, data), dim=1)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
    
def train_and_save_model(loaders, model: nn.Module, criterion:nn.Module, optimizer:torch.optim.Optimizer, scheduler, device, num_epochs):
    min_loss = None
    # model.load_state_dict(state_dict=torch.load(os.path.join('Utils', 'best_model_params.pt')), strict=True)

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            sz = 0

            for inputs, data, targets in tqdm(loaders[phase]):
                sz += inputs.size(0)
                inputs = inputs.to(device)
                targets = targets.to(device)
                data = data.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, data)
                    loss = criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = epoch_loss / sz

            print(f'{phase} Loss at Epoch {epoch}: {epoch_loss}')

            if phase == 'val':
                if min_loss is None or min_loss > epoch_loss:
                    min_loss = epoch_loss
                    torch.save(model.state_dict(), os.path.join('Utils', 'best_model_params.pt'))

def load_and_test_model(loaders, model: nn.Module, criterion, device):
    model.load_state_dict(state_dict=torch.load(os.path.join('Utils', 'best_model_params.pt')), strict=True)
    model.eval()
    loader = loaders['test']

    metrics = {
        'Accuracy': Accuracy(task='binary').to(device),
        'Precision': Precision(task='binary').to(device), 
        'Recall': Recall(task='binary').to(device), 
        'F1 score': F1Score(task='binary').to(device),
        'AUC ROC': AUROC(task='binary').to(device)
        }

    sz, loss = 0, 0.0
    for inputs, data, targets in tqdm(loader):
        sz += inputs.size(0)
        inputs = inputs.to(device)
        data = data.to(device)
        targets = targets.to(device)
        
        with torch.set_grad_enabled(False):
            outputs = model(inputs, data)
            loss += criterion(outputs, targets).item() * inputs.size(0)

            for name, metric in metrics.items():
                metric(outputs, targets)

    loss = loss / sz

    print(f'Loaded model\'s Loss: {loss}')
    print('-'*10)
    for name, metric in metrics.items():
        print(f'{name}: {metric.compute()}')

def train_init(device, epochs=50, lr=(0.01, 0.001), batch_sz=64):
    dataset = AudioDataset(tsv_path=os.path.join('Utils', 'targets.tsv'), type='train')
    targets = dataset.targets['Label'].to_numpy()
    train_val_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, stratify=targets, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, stratify=targets[train_val_idx], random_state=42)
    
    train_sampler, val_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx), SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_sz, sampler=train_sampler)
    val_loader = DataLoader(dataset=dataset, batch_size=batch_sz, sampler=val_sampler)
    test_loader = DataLoader(dataset=dataset, batch_size=batch_sz, sampler=test_sampler)

    loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    model = CNNModel().to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)

    optimizer = torch.optim.Adam(params=[{'params': model.pretrained.parameters(), 'lr': lr[1]}, 
                                         {'params': model.fc1.parameters()},
                                         {'params': model.fc2.parameters()},
                                         {'params': model.fc3.parameters()}
                                         ], lr=lr[0])

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.97)

    # train_and_save_model(loaders, model, criterion, optimizer, lr_scheduler, device, num_epochs=epochs)
    load_and_test_model(loaders, model, criterion, device)

def predict_gender():
    model = CNNModel()
    model.load_state_dict(torch.load(os.path.join('Utils', 'best_model_params.pt')))
    model.eval()

    img_size=(224, 224)
    sigm = nn.Sigmoid()
    
    features = pd.read_csv(os.path.join('Utils', f'test_features.csv'), index_col=False, header=0)

    predictions = []
    names = []
    cnt = 0
    for file in tqdm(os.listdir(os.path.join('Data', f'test_img')), disable=False):
        name, _ = os.path.splitext(os.fsdecode(file))
        names.append(name)
        path = os.path.join('Data', f'test_img', name + '.png')
        
        img = decode_image(input=path, mode=torchvision.io.ImageReadMode.RGB)
        
        data = torch.tensor(np.array(features[features['name'] == name].values[0, 1:], dtype=np.float32))
        data = data.unsqueeze(dim=0)

        transforms = v2.Compose([
            v2.Resize(size=img_size), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.1953, 0.0678, 0.2199], std=[0.2656, 0.1031, 0.2017])
            ])
        img: torch.Tensor = transforms.forward(img)
        img = img.unsqueeze(dim=0)
        
        pred = model.forward(img, data)
        pred = sigm(pred).item()

        pred = int(pred > 0.5)
        predictions.append(pred)
    
    predictions = pd.DataFrame({'Name': names, 'Label': predictions})
    predictions.to_csv(os.path.join('Utils', 'predictions.tsv'), sep='\t', header=False, index=False)

if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('CUDA in use...')
        device = torch.device('cuda')

    # train_init(device)
    # predict_gender()

    df: pd.DataFrame = pd.read_csv(os.path.join('Utils', 'predictions.tsv'), sep='\t', names=['Name', 'Label'])
    print(df['Label'].value_counts())

    df: pd.DataFrame = pd.read_csv(os.path.join('Utils', 'targets.tsv'), sep='\t', names=['Name', 'Label'])
    print(df['Label'].value_counts())
