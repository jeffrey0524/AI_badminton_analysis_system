from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import efficientnet
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import math
from tqdm import tqdm
import dataprocess

from torchvision import models

config = {
    'valid_ratio': 0,   # validation_size = train_size * valid_ratio
    'n_epochs': 30,     # Number of epochs.            
    'batch_size': 1, 
    'learning_rate': 1e-5,              
    'early_stop': 20,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': PROJECT_DIR/'src/models/model_2.ckpt'  # Your model will be saved here.
    }  

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        # find the feature
        print("Module_init")
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(1, 1, 11)).to(device).type(torch.cuda.FloatTensor)

        self.features = efficientnet.efficientnet_b1(weights = models.EfficientNet_B1_Weights.IMAGENET1K_V1) #! change to recommended 
        # models.EfficientNet_B1_Weights.IMAGENET1K_V1 | models.EfficientNet_B1_Weights.DEFAULT

    def forward(self, x):
        x = x.permute(0, 3, 4, 1, 2)
        x = x.to(device).type(torch.cuda.FloatTensor)
        x = self.conv3d(x)

        x = self.features(x)
        x[:, :14] = nn.Softmax()(x[:, :14])
        x[:, 15:20] = nn.Sigmoid()(x[:, 15:20])
        x[:, 21:26] = nn.ReLU()(x[:, 21:26])
        x[:, 27:38] = nn.Softmax()(x[:, 27:38])
        return x
        
def trainer(train_data_list, valid_data_list, model, config, device):
    loss_func1 = nn.CrossEntropyLoss()
    loss_func2 = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) 

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    writer = SummaryWriter()
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        print("checkpoint1")

        # total loss
        for train_data in DataLoader(train_data_list, batch_size=config['batch_size'], shuffle=True):
            x = torch.Tensor(dataprocess.list_to_cube([[train_data[0][0],int(train_data[1][0])]]))
            y = train_data[2]

            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            print('checkpoint1.5')
            pred = model(x)
            print("checkpoint2")
            loss_frame = loss_func1(pred[:, :11], y)
            loss_hitter = loss_func1(pred[:, 12:14], y)
            loss_roundhead = loss_func2(pred[:, 15:16], y)
            loss_backhand = loss_func2(pred[:, 17:18], y)
            loss_ballheight = loss_func2(pred[:, 19:20], y)
            loss_landing_X = loss_func1(pred[:, 21], y)
            loss_landing_Y = loss_func1(pred[:, 22], y)
            loss_Hitterland_X = loss_func1(pred[:, 23], y)
            loss_Hitterland_Y = loss_func1(pred[:, 24], y)
            loss_Defender_X = loss_func1(pred[:, 25], y)
            loss_Defender_Y = loss_func1(pred[:, 26], y)
            loss_balltype = loss_func1(pred[:, 27:35], y)
            loss_winner = loss_func1(pred[:, 36:38], y)

            total_Loss = 0.5*loss_frame 
            + 0.03125*(loss_roundhead + loss_backhand + loss_Hitterland_X + loss_Hitterland_Y
                        + loss_Defender_X + loss_Defender_Y)
            + 0.0625*(loss_hitter + loss_ballheight + loss_landing_X + loss_landing_Y + loss_winner)
            + 0.125*(loss_balltype)

            total_Loss.backward()
            optimizer.step()
            step += 1

            loss_record.append(total_Loss.detach().item())

            mean_train_loss = sum(loss_record)/len(loss_record)
            writer.add_scalar('Loss/train', mean_train_loss, step)
        # valiation
        model.eval()
        loss_record = []
        for valid_data in DataLoader(valid_data_list, batch_size=config['batch_size'], shuffle=True):
            x = torch.Tensor(dataprocess.list_to_cube([[valid_data[0][0],int(valid_data[1][0])]]))
            y = valid_data[2]

            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss_frame = loss_func1(pred[:, :11], y)
                loss_hitter = loss_func1(pred[:, 12:14], y)
                loss_roundhead = loss_func2(pred[:, 15:16], y)
                loss_backhand = loss_func2(pred[:, 17:18], y)
                loss_ballheight = loss_func2(pred[:, 19:20], y)
                loss_landing_X = loss_func1(pred[:, 21], y)
                loss_landing_Y = loss_func1(pred[:, 22], y)
                loss_Hitterland_X = loss_func1(pred[:, 23], y)
                loss_Hitterland_Y = loss_func1(pred[:, 24], y)
                loss_Defender_X = loss_func1(pred[:, 25], y)
                loss_Defender_Y = loss_func1(pred[:, 26], y)
                loss_balltype = loss_func1(pred[:, 27:35], y)
                loss_winner = loss_func1(pred[:, 36:38], y)

                total_Loss = 0.5*loss_frame 
                + 0.03125*(loss_roundhead + loss_backhand + loss_Hitterland_X + loss_Hitterland_Y
                        + loss_Defender_X + loss_Defender_Y)
                + 0.0625*(loss_hitter + loss_ballheight + loss_landing_X + loss_landing_Y + loss_winner)
                + 0.125*(loss_balltype)
                loss_record.append(total_Loss.item())
        
            mean_valid_loss = sum(loss_record)/len(loss_record)
            print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
            writer.add_scalar('Loss/valid', mean_valid_loss, step)

            if mean_valid_loss < best_loss:
                best_loss = mean_valid_loss
                torch.save(model.state_dict(), config['save_path']) # Save your best model
                print('Saving model with loss {:.3f}...'.format(best_loss))
                early_stop_count = 0
            else: 
                early_stop_count += 1

            if early_stop_count >= config['early_stop']:
                print('\nModel is not improving, so we halt the training session.')
                return

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_list = dataprocess.get_data_list('../test_part1/train/')
train_data_list, valid_data_list = train_test_split(data_list, test_size=0.1, random_state=42)
trainer(train_data_list, valid_data_list, Module(), config, device)