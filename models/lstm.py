import torch
from torch import nn
import torchvision.transforms as transforms
# __all__ = ['lstm']
torch.manual_seed(1)


class CharLSTM(nn.Module):
    def __init__(self, args):
        super(CharLSTM, self).__init__()
        self.args = args
        self.embed = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, 2, batch_first=True, dropout=0.5)
        # self.drop = nn.Dropout()
        self.n_hidden = 256
        self.fc = nn.Linear(256, 80)

        # def forward(self, x):
        #     x = self.embed(x)
        #     x, hidden = self.lstm(x)
        #     x = self.drop(x)
        #     return self.out(x[:, -1, :])
    def forward(self, x, out_activation=False):
            x_ = self.embed(x)
            h0 = torch.rand(2, x_.size(0), self.n_hidden).to(self.args.device)
            c0 = torch.rand(2, x_.size(0), self.n_hidden).to(self.args.device)
            activation, (h_n, c_n) = self.lstm(x_, (h0, c0))

            fc_ = activation[:, -1, :]

            output = self.fc(fc_)
            if out_activation:
                return output, activation
            else:
                return output


class ModelLSTMShakespeare(nn.Module):
    def __init__(self):
        super(ModelLSTMShakespeare, self).__init__()
        self.embedding_len = 8
        self.seq_len = 80
        self.num_classes = 80
        self.n_hidden = 256
        # self.batch_size = batch_size_train

        self.embeds = nn.Embedding(self.seq_len, self.embedding_len)
        self.multi_lstm = nn.LSTM(input_size=self.embedding_len, hidden_size=self.n_hidden,
                                  num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(self.n_hidden, self.num_classes)

    def forward(self, x, out_activation=False):
        x = x.to(torch.int64)
        x_ = self.embeds(x)
        h0 = torch.rand(2, x_.size(0), self.n_hidden).to(torch.device('cuda:0'))
        c0 = torch.rand(2, x_.size(0), self.n_hidden).to(torch.device('cuda:0'))
        activation, (h_n, c_n) = self.multi_lstm(x_,(h0,c0))

        fc_ = activation[:, -1, :]

        output = self.fc(fc_)
        if out_activation:
            return output, activation
        else:
            return output
