import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import Sampling_train
from utils import LabeltoStr

# torch.cuda.set_device(0)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(420, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, x):
        # print(np.shape(x))  # torch.Size([64, 3, 140, 440])

        x = x.reshape(-1, 420, 440).permute(0, 2, 1)
        # print(np.shape(x))  # torch.Size([64, 440, 420])

        x = x.reshape(-1, 420)
        # print(np.shape(x))  # torch.Size([28160, 420])

        fc1 = self.fc1(x)
        # print(np.shape(fc1))  # torch.Size([28160, 128])

        fc1 = fc1.reshape(-1, 440, 128)
        # print(np.shape(fc1))  # torch.Size([64, 440, 128])

        lstm, (h_n, h_c) = self.lstm(fc1, None)
        # print(np.shape(lstm))  # torch.Size([64, 440, 256])

        out = lstm[:, -1, :]
        # print(np.shape(out))  # torch.Size([64, 256])

        return out


class Decoder(nn.Module):
    def __init__(self, bidirectional=True):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)

        if bidirectional == True:
            self.out = nn.Linear(128*2, 65)
        else:
            self.out = nn.Linear(128, 65)  # 定义全连接层

    def forward(self, x):
        # print(np.shape(x))  # torch.Size([64, 256])

        x = x.reshape(-1, 1, 256)
        # print(np.shape(x))  # torch.Size([64, 1, 256])

        x = x.expand(-1, 7, 256)
        # print(np.shape(x))  # torch.Size([64, 7, 256])

        lstm, (h_n, h_c) = self.lstm(x, None)
        # print(np.shape(lstm))  # torch.Size([64, 7, 256])

        y1 = lstm.reshape(-1, 128*2)
        # print(np.shape(y1))  # torch.Size([448, 256])

        out = self.out(y1)
        # print(np.shape(out))  # torch.Size([448, 65])

        output = out.reshape(-1, 7, 65)  # 10表示输出十个值，可以更改
        # print(np.shape(output))  # torch.Size([64, 4, 65])  64

        return output


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)

        return decoder


if __name__ == '__main__':
    BATCH = 64
    EPOCH = 10000
    save_path = r'params/seq2seq.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MainNet().to(device)

    opt = torch.optim.Adam(net.parameters())
    loss_func = nn.MSELoss()

    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    else:
        print("No Params!")

    train_data = Sampling_train.Sampling(root="./blue_plate")
    valida_data = Sampling_train.Sampling(root="./blue_plate2")
    train_loader = data.DataLoader(dataset=train_data,
                                   batch_size=BATCH, shuffle=True)
    valida_loader = data.DataLoader(dataset=valida_data, batch_size=64, shuffle=True)

    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            # print(np.shape(x))  # torch.Size([384, 3, 224, 224])
            # print(np.shape(y))  # torch.Size([64, 7, 65])

            batch_x = x.to(device)
            batch_y = y.float().to(device)

            output = net(batch_x)
            # print(np.shape(output))  # torch.Size([64, 7, 65])

            loss = loss_func(output, batch_y)
            # print(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 10 == 0:
                label_y = torch.argmax(y, 2).detach().numpy()
                # print(label_y)

                out_y = torch.argmax(output, 2).cpu().detach().numpy()
                # print(out_y)
                # print(np.sum(out_y == label_y, dtype=np.float32))

                accuracy = np.sum(
                    out_y == label_y, dtype=np.float32) / (BATCH * 4)
                print("epoch:{},i:{},loss:{:.4f},acc:{:.2f}%"
                      .format(epoch, i, loss.item(), accuracy * 100))

                # print("label_y:", LabeltoStr(label_y[0]))
                # print("out_y:", LabeltoStr(out_y[0]))

        torch.save(net.state_dict(), save_path)

        for i, (x, y) in enumerate(valida_loader):
            # print(np.shape(x))  # torch.Size([64, 3, 60, 240])
            # print(np.shape(y))  # torch.Size([64, 4, 62])

            batch_x = x.to(device)
            batch_y = y.float().to(device)

            output = net(batch_x)
            # print(np.shape(output))  # torch.Size([64, 4, 62])

            loss = loss_func(output, batch_y)
            # print(loss.item())

            if i % 2 == 0:
                label_y = torch.argmax(y, 2).detach().numpy()
                # print(label_y)

                out_y = torch.argmax(output, 2).cpu().detach().numpy()
                # print(out_y)
                # print(np.sum(out_y == label_y, dtype=np.float32))

                accuracy = np.sum(
                    out_y == label_y, dtype=np.float32) / (BATCH * 4)
                print("valida_epoch:{},i:{},loss:{:.4f},acc:{:.2f}%"
                      .format(epoch, i, loss.item(), accuracy * 100))

                print("label_y:", LabeltoStr(label_y[0]))
                print("out_y:", LabeltoStr(out_y[0]))
