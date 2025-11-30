import numpy as np
# from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
# from torchsummary import summary
import torch.utils.data as Data
import os


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(2, 64, 7, 2, 3),  # [, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 64, 7, 2, 3),  # [, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 64, 5, 2, 2),  # [, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.Conv2d(64, 128, 5, 2, 2),  # [, 64, 8, 8]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, 5, 2, 2),  # [, 64, 4, 4]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 128, 2, 2, 0),  # [, 64, 4, 4]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 64, 2, 1, 0),  # [, 64, 1, 1]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 2, 1, 0),  # [, 128, 24, 24]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 128, 2, 2, 0),  # [, 128, 24, 24]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 128, 6, 2, 2),  # [, 128, 48, 48]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, 6, 2, 2),  # [, 64, 48, 48]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 64, 6, 2, 2),  # [, 32, 48, 48]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 64, 8, 2, 3),  # [, 32, 48, 48]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 2, 8, 2, 3)

        )
    '''
    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder
    '''

    def forward(self, x):
        encoder = self.encoder(x)
        return encoder

def normalize(data):
    '''
    mu=np.mean(data)
    sigma=np.std(data)
    print(mu,sigma)
    '''
    mask = np.zeros([1000, 2, 128, 128])
    mask[:, :, 4:124, 4:124] = 1
    data[mask == 0] = 0
    '''
    mu=np.mean(data)
    sigma=np.std(data)
    print(mu,sigma)
    '''
    mu = 0
    sigma = 1.0

    return (data - mu) / sigma


if __name__ == '__main__':
    model = AutoEncoder()
    model = model.cuda()
    criterion = nn.MSELoss().cuda()
    lr = 1e-3
    # summary(model, (2, 128, 128))
    path = '/home/msi/yxh/DATA/scoop_data_final'
    epoches = 40
    optimizer = optim.Adam(model.parameters(), lr=lr)

    files = os.listdir(path)
    for epoch in range(epoches):
        print(epoch)
        if epoch in [24, 35]:
            lr *= 0.1
        test_loss_epoch = 0
        test_num = 0
        for file in files:
            train_dataset = np.load(path + '/' + file).reshape([-1, 2, 128, 128])
            train_dataset = normalize(train_dataset)
            train_data, test_data = torch.utils.data.random_split(train_dataset, [896, 104])
            train_loader = Data.DataLoader(
                dataset=train_data,
                batch_size=128,
                shuffle=True,
                num_workers=4
            )
            test_loader = Data.DataLoader(
                dataset=test_data,
                batch_size=128,
                shuffle=True,
                num_workers=4
            )
            model.train()
            train_loss_epoch = 0
            train_num = 0

            for step, img in enumerate(train_loader):
                img = img.cuda().float()
                _, output = model(img)
                output = output.cuda()
                loss = criterion(output, img)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item() * img.size(0)
                train_num += img.size(0)
                print('epoch:{} loss:{:7f}'.format(epoch, loss.item()))

            model.eval()

            for step, img in enumerate(test_loader):
                img = img.cuda().float()
                _, output = model(img)
                output = output.cuda()
                loss = criterion(output, img)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                test_loss_epoch += loss.item() * img.size(0)
                test_num += img.size(0)
        
        print('epoch:{} test_loss:{:7f}'.format(epoch, l))

        torch.save(model, "./scoop_model_final/autoencoder%d.pkl" % epoch)