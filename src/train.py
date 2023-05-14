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



def vaild_spilt(dataset, valid_ratio):
    vaild_set_size = int(valid_ratio * len(dataset))
    train_set_size = len(dataset) - vaild_set_size
    train_set, vaild_set = random_split(dataset, [train_set_size, vaild_set_size])
    return np.array(train_set), np.array(vaild_set)

class BadmintonDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
    
    def __len__(self):
        return len(self.data)

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        # find the feature
        self.features = efficientnet.efficientnet_b1(pretrained=True)   

    def forward(self, x):
        x = self.features(x)
        x[:, :14] = nn.Softmax()(x[:, :14])
        x[:, 15:20] = nn.Sigmoid()(x[:, 15:20])
        x[:, 21:26] = nn.ReLU()(x[:, 21:26])
        x[:, 27:38] = nn.Softmax()(x[:, 27:38])
        return x
        
def trainer(train_loader, valid_loader, model, config, device):
    loss_func1 = nn.CrossEntropyLoss
    loss_func2 = nn.BCELoss

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) 

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    writer = SummaryWriter()
    for epoch in range(n_epochs):
        model.train()
        loss_record = []

        # total loss
        for x, y in train_loader:
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

            total_Loss.backward()
            optimizer.step()
            step += 1

            loss_record.append(total_Loss.detach().item())

            mean_train_loss = sum(loss_record)/len(loss_record)
            writer.add_scalar('Loss/train', mean_train_loss, step)
        # valiation
        model.eval()
        loss_record = []
        for x, y in valid_loader:
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

config = {
    'valid_ratio': 0,   # validation_size = train_size * valid_ratio
    'n_epochs': 100,     # Number of epochs.            
    'batch_size': 100, 
    'learning_rate': 1e-5,              
    'early_stop': 20,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
with open('/home/aivc2/AI_bedminton_dataset/src/pickle/cube.pickle', 'rb') as file:
    train_data = pickle.load(file)

data = dataprocess.get_data_list('../test_part1/train/')
labels = []
for i in data:
    labels.append(i[2])

print(train_data.shape)
# with open('test_data.pickle', 'rb') as file:
#     test_data = pickle.load(file)

train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, labels, test_size=0.1, random_state=42)

# print(f"""train_data size: {train_data.shape} 
# valid_data size: {valid_data.shape} 
# """)

train_dataset , valid_dataset = BadmintonDataset(train_data, labels),\
                                BadmintonDataset(valid_data, labels)




train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
# test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

model = Module()
print(train_loader,  valid_loader)
trainer(train_loader, valid_loader, model, config, device)
