import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import efficientnet
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
import math
from tqdm import tqdm


def vaild_spilt(dataset, valid_ratio):
    vaild_set_size = int(valid_ratio * len(dataset))
    train_set_size = len(dataset) - vaild_set_size
    train_set, vaild_set = random_split(dataset, [train_set_size, vaild_set_size])
    return np.array(train_set), np.array(vaild_set)

class BadmintonDataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        # find the feature
        self.features = efficientnet.efficientnet_b1(pretrained=True)

        self.acc_frame = nn.Sequential(
            nn.Linear(1280, 1),
            nn.ReLU()
        )

        self.hitter_land_X = nn.Sequential(
            nn.Linear(1280, 1),
            nn.ReLU()
        )
        self.hitter_land_Y = nn.Sequential(
            nn.Linear(1280, 1),
            nn.ReLU()
        )
        self.defender_land_X = nn.Sequential(
            nn.Linear(1280, 1),
            nn.ReLU()
        )
        self.defender_land_Y = nn.Sequential(
            nn.Linear(1280, 1),
            nn.ReLU()
        )
        self.ball_land_X = nn.Sequential(
            nn.Linear(1280, 1),
            nn.ReLU()
        )
        self.ball_land_Y = nn.Sequential(
            nn.Linear(1280, 1),
            nn.ReLU()
        )

        self.acc_hitter = nn.Sequential(
            nn.Linear(1280, 3),
            nn.Softmax(dim=1)
        )

        self.acc_winner = nn.Sequential(
            nn.Linear(1280, 3),
            nn.Softmax(dim=1)
        )

        self.backhand = nn.Sequential(
            nn.Linear(1280, 2),
            nn.Sigmoid()
        )
        self.ballheight = nn.Sequential(
            nn.Linear(1280, 2),
            nn.Sigmoid()
        )
        self.roundhead = nn.Sequential(
            nn.Linear(1280, 2),
            nn.Sigmoid()
        )

        self.balltypes = nn.Sequential(
            nn.Linear(1280, 9),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)

        frame_output = self.acc_frame(x)
        hitter_output = self.acc_hitter(x)
        roundhead_output = self.roundhead(x)
        backhand_output = self.backhand(x)
        ballheight_output = self.ballheight(x)

        ball_land_output_X = self.ball_land_X(x)
        ball_land_output_Y = self.ball_land_Y(x)
        hitter_land_X_output = self.hitter_land_X(x)
        hitter_land_Y_output = self.hitter_land_Y(x)
        defender_land_X_output = self.defender_land_X(x)
        defender_land_Y_output = self.defender_land_Y(x)

        balltypes_output = self.balltypes(x)
        winner_output = self.acc_winner(x)  


        return frame_output, hitter_output, roundhead_output, backhand_output, ballheight_output,\
                ball_land_output_X, ball_land_output_Y, hitter_land_X_output, hitter_land_Y_output,\
                defender_land_X_output, defender_land_Y_output, balltypes_output, winner_output
        
    def trainer(train_loader, valid_loader, model, config, device):
        loss_func1 = nn.CrossEntropyLoss()
        loss_func2 = nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], momentum=0.9) 

        n_epochs, best_loss, step, early_stop_counter = config['n_epochs'], math.inf, 0, 0

        writer = SummaryWriter()
        for epoch in range(n_epochs):
            model.train()
            loss_record = []

            train_pbar = tqdm(train_loader, position=0, leave=True)

            # total loss
            for x, y in train_pbar:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss_frame = loss_func1(pred[0], y[0])
                loss_hitter = loss_func1(pred[1], y[1])
                loss_roundhead = loss_func2(pred[2], y[2])
                loss_backhand = loss_func2(pred[3], y[3])
                loss_ballheight = loss_func2(pred[4], y[4])
                loss_landing_X = loss_func1(pred[5], y[5])
                loss_landing_Y = loss_func1(pred[6], y[6])
                loss_Hitterland_X = loss_func1(pred[7], y[7])
                loss_Hitterland_Y = loss_func1(pred[8], y[8])
                loss_Defender_X = loss_func1(pred[9], y[9])
                loss_Defender_Y = loss_func1(pred[10], y[10])
                loss_balltype = loss_func1(pred(11), y[11])
                loss_winner = loss_func1(pred[12], y(12))

                total_Loss = 0.5*loss_frame 
                + 0.03125*(loss_roundhead + loss_backhand + loss_Hitterland_X + loss_Hitterland_Y
                          + loss_Defender_X + loss_Defender_Y)
                + 0.0625*(loss_hitter + loss_ballheight + loss_landing_X + loss_landing_Y + loss_winner)
                + 0.125*(loss_balltype)

                total_Loss.backward()
                optimizer.step()
                step += 1

                loss_record.append(total_Loss.detach().item())

                train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
                train_pbar.set_postfix({'loss': total_Loss.detach().item()})

                mean_train_loss = sum(loss_record)/len(loss_record)
                writer.add_scalar('Loss/train', mean_train_loss, step)
            # valiation
            model.eval()
            loss_record = []
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    pred = model(x)
                    loss_frame = loss_func1(pred[0], y[0])
                    loss_hitter = loss_func1(pred[1], y[1])
                    loss_roundhead = loss_func2(pred[2], y[2])
                    loss_backhand = loss_func2(pred[3], y[3])
                    loss_ballheight = loss_func2(pred[4], y[4])
                    loss_landing_X = loss_func1(pred[5], y[5])
                    loss_landing_Y = loss_func1(pred[6], y[6])
                    loss_Hitterland_X = loss_func1(pred[7], y[7])
                    loss_Hitterland_Y = loss_func1(pred[8], y[8])
                    loss_Defender_X = loss_func1(pred[9], y[9])
                    loss_Defender_Y = loss_func1(pred[10], y[10])
                    loss_balltype = loss_func1(pred(11), y[11])
                    loss_winner = loss_func1(pred[12], y(12))

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
    'valid_ratio': 0.1,   # validation_size = train_size * valid_ratio
    'n_epochs': 3000,     # Number of epochs.            
    'batch_size': 256, 
    'learning_rate': 1e-5,              
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
with open('/home/aivc2/AI_bedminton_dataset/src/pickle/cube.pickle', 'rb') as file:
    train_data = pickle.load(file)

print(train_data.shape)
# with open('test_data.pickle', 'rb') as file:
#     test_data = pickle.load(file)

train_data, valid_data = vaild_spilt(train_data, config['valid_ratio'])

print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
""")

train_dataset , valid_dataset = BadmintonDataset(train_data),\
                                BadmintonDataset(valid_data)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
# test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

model = Module()

