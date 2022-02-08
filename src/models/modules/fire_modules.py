import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from src.models.modules.convlstm import ConvLSTM
from torchvision.models.resnet import resnet18
from fastai.vision.models import unet

np.seterr(divide='ignore', invalid='ignore')


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=1):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


# class SimpleConvLSTM(nn.Module):
#     def __init__(self, hparams: dict):
#         super().__init__()
#
#         # lstm part
#         input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
#         hidden_size = hparams['hidden_size']
#         lstm_layers = hparams['lstm_layers']
#         # clstm part
#         self.convlstm = ConvLSTM(input_dim,
#                                  hidden_size,
#                                  (3, 3),
#                                  lstm_layers,
#                                  True,
#                                  True,
#                                  False)
#
#         # m = resnet18(pretrained=False)
#         # m.conv1 = nn.Conv2d(hidden_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # num_ftrs = m.fc.in_features
#         # m.fc = nn.Linear(num_ftrs, 2)
#         # self.m = m
#         # cnn part
#         # self.conv1 = nn.Conv2d(hidden_size, 8, kernel_size=3, stride=1, padding=1)
#         # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
#         # self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
#         # fully-connected part
#
#         kernel_size = 3
#         self.ln1 = torch.nn.LayerNorm(input_dim)
#         # self.se1 = SE_Block(hidden_size, r=4)
#         self.conv1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=1, padding=1)
#         # self.se2 = SE_Block(hidden_size, r=4)
#         self.drop1 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(kernel_size * kernel_size * hidden_size * 16, 16)
#         # self.fc1 = nn.Linear(10000, 16)
#         self.drop2 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(16, 8)
#         self.drop3 = nn.Dropout(0.5)
#         self.fc3 = nn.Linear(8, 2)

# Pyramid dilated convlstm
class PDConvLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # lstm part
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        # clstm part
        self.convlstm1 = ConvLSTM(input_dim,
                                  hidden_size,
                                  (3, 3),
                                  lstm_layers,
                                  True,
                                  True,
                                  False, dilation=1)
        self.convlstm2 = ConvLSTM(input_dim,
                                  hidden_size,
                                  (3, 3),
                                  lstm_layers,
                                  True,
                                  True,
                                  False, dilation=2)

        # m = resnet18(pretrained=False)
        # m.conv1 = nn.Conv2d(hidden_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # num_ftrs = m.fc.in_features
        # m.fc = nn.Linear(num_ftrs, 2)
        # self.m = m
        # cnn part
        # self.conv1 = nn.Conv2d(hidden_size, 8, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        # fully-connected part

        kernel_size = 3
        self.ln1 = torch.nn.LayerNorm(input_dim)
        # self.se1 = SE_Block(hidden_size, r=4)
        self.conv1 = nn.Conv2d(2 * hidden_size, hidden_size, kernel_size=kernel_size, stride=1, padding=1)
        # self.se2 = SE_Block(hidden_size, r=4)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear((25 // 2) * (25 // 2) * hidden_size, hidden_size)
        # self.fc1 = nn.Linear(10000, 16)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.drop3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size // 2, 2)

    def forward(self, x: torch.Tensor):
        # (b x t x c x h x w) -> (b x t x h x w x c) -> (b x t x c x h x w)
        x = self.ln1(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        _, last_states1 = self.convlstm1(x)
        x1 = last_states1[0][0]
        _, last_states2 = self.convlstm2(x)
        x2 = last_states2[0][0]
        x = torch.cat([x1, x2], 1)
        # x = self.se1(x)

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # x = self.se2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop1(self.fc2(x)))
        x = self.fc3(x)
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        # x = torch.flatten(x, 1)
        # x = F.relu(self.drop1(self.fc1(x)))
        # x = F.relu(self.drop2(self.fc2(x)))
        return torch.nn.functional.log_softmax(x, dim=1)


class SimpleConvLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        # clstm part
        self.convlstm = ConvLSTM(input_dim,
                                 hidden_size,
                                 (3, 3),
                                 lstm_layers,
                                 True,
                                 True,
                                 False, dilation=1)

        kernel_size = 3
        self.ln1 = torch.nn.LayerNorm(input_dim)
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(kernel_size, kernel_size), stride=(1, 1),
                               padding=(1, 1))
        # fully-connected part
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear((25 // 2) * (25 // 2) * hidden_size, hidden_size)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 2)

    def forward(self, x: torch.Tensor):
        # (b x t x c x h x w) -> (b x t x h x w x c) -> (b x t x c x h x w)
        x = self.ln1(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        _, last_states = self.convlstm(x)
        x = last_states[0][0]
        # cnn
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # fully-connected
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class SimpleLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # lstm part
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        self.ln1 = torch.nn.LayerNorm(input_dim)
        self.lstm = torch.nn.LSTM(input_dim, hidden_size, num_layers=lstm_layers, batch_first=True)
        # fully-connected part
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.drop1 = torch.nn.Dropout(0.5)

        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size // 2, hidden_size // 4)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc3 = torch.nn.Linear(hidden_size // 4, 2)

        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            self.relu,
            self.drop1,
            self.fc2,
            self.relu,
            self.drop2,
            self.fc3
        )

    def forward(self, x: torch.Tensor):
        x = self.ln1(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc_nn(lstm_out[:, -1, :])
        return torch.nn.functional.log_softmax(x, dim=1)


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths=None):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        if lengths:
            for i, l in enumerate(lengths):  # skip the first sentence
                if l < max_len:
                    mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class SimpleLSTMAttention(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # lstm part
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        hidden_dim = hidden_size
        lstm_layers = hparams['lstm_layers']
        lstm_layer = 2
        self.dropout = nn.Dropout(p=0.5)
        self.lstm1 = nn.LSTM(input_size=input_dim,
                             hidden_size=hidden_size,
                             num_layers=lstm_layers,
                             bidirectional=True)
        self.atten1 = Attention(hidden_dim * 2, batch_first=True)  # 2 is bidrectional
        self.lstm2 = nn.LSTM(input_size=hidden_dim * 2,
                             hidden_size=hidden_dim,
                             num_layers=1,
                             bidirectional=True)
        self.atten2 = Attention(hidden_dim * 2, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(hidden_dim * lstm_layer * 2, hidden_dim * lstm_layer * 2),
                                 nn.BatchNorm1d(hidden_dim * lstm_layer * 2),
                                 nn.ReLU())
        self.fc2 = nn.Linear(hidden_dim * lstm_layer * 2, 2)

    def forward(self, x):
        out1, (h_n, c_n) = self.lstm1(x)
        x, _ = self.atten1(out1)  # skip connect

        out2, (h_n, c_n) = self.lstm2(out1)
        y, _ = self.atten2(out2)

        z = torch.cat([x, y], dim=1)
        z = self.fc1(self.dropout(z))
        z = self.fc2(self.dropout(z))
        return torch.nn.functional.log_softmax(z, dim=1)


class SK_CLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        self.convlstm1 = ConvLSTM(input_dim, hidden_size, (3, 3), 1, True, True, False, dilation=1)
        self.convlstm2 = ConvLSTM(input_dim, hidden_size, (3, 3), 1, True, True, False, dilation=2)
        self.fc1 = nn.Linear(2 * hidden_size * 25 * 25, hidden_size)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size // 2, 2)

    def forward(self, x):
        _, last_states1 = self.convlstm1(x)
        _, last_states2 = self.convlstm2(x)
        x = F.relu(torch.cat([last_states1[0][0], last_states2[0][0]], 1))
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class SK_LSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        self.lstm = nn.LSTM(input_dim, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.drop1 = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size // 2, hidden_size // 4)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc3 = torch.nn.Linear(hidden_size // 4, 2)
        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            self.drop1,
            self.relu,
            self.fc2,
            self.drop2,
            self.relu,
            self.fc3
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc_nn(lstm_out[:, -1, :])
        return torch.nn.functional.log_softmax(x, dim=1)


class SK_CNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        kernel_size = 3
        self.conv1 = nn.Conv2d(input_dim, hidden_size, kernel_size=kernel_size, stride=1, padding=1)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(kernel_size * kernel_size * hidden_size * 16, 16)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 8)
        self.drop3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop1(self.fc2(x)))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class SimpleCNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # CNN definition
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        self.conv1 = nn.Conv2d(input_dim, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(576, 64)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 32)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        return torch.nn.functional.log_softmax(self.fc3(x), dim=1)


class SimpleFCN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        # CNN definition
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        self.conv1 = nn.Conv2d(input_dim, 8, kernel_size=(6, 6), stride=(1, 1), padding=(1, 1))
        self.conv11 = nn.Conv2d(8, 8, kernel_size=(7, 7), stride=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(9, 9), stride=(1, 1), padding=(1, 1))
        self.conv22 = nn.Conv2d(16, 16, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
        self.conv33 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        k = 4
        self.fconv = nn.Conv2d(16, k * k * 16, kernel_size=(k, k), stride=(1, 1), padding=(0, 0))
        self.conv1_1x1 = nn.Conv2d(k * k * 16, 64, (1, 1))
        self.drop1 = nn.Dropout(0.5)
        self.conv2_1x1 = nn.Conv2d(64, 32, (1, 1))
        self.drop2 = nn.Dropout(0.5)
        self.conv3_1x1 = nn.Conv2d(32, 2, (1, 1))

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv22(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv33(x))
        print(x.shape)
        x = self.fconv(x)
        print(x.shape)

        x = F.relu(self.drop1(self.conv1_1x1(x)))
        print(x.shape)

        x = F.relu(self.drop2(self.conv2_1x1(x)))
        print(x.shape)

        return torch.nn.functional.log_softmax(self.conv3_1x1(x), dim=1)


class DynUnet(nn.Module):
    def __init__(self, hparams: dict):
        super(DynUnet, self).__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        m = resnet18()
        m = nn.Sequential(*list(m.children())[:-2])
        m[0] = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net = unet.DynamicUnet(m, 2, (25, 25), norm_type=None)

    def forward(self, x):
        out = self.net(x)
        return torch.nn.functional.log_softmax(out, dim=1)


class DynUnetConvLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super(DynUnetConvLSTM, self).__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        self.convlstm = ConvLSTM(input_dim, hidden_size, (3, 3), 1, True, True, False)
        self.se1 = SE_Block(hidden_size, r=input_dim)
        m = resnet18()
        m = nn.Sequential(*list(m.children())[:-2])
        m[0] = nn.Conv2d(hidden_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net = unet.DynamicUnet(m, 2, (65, 65), norm_type=None)

    def forward(self, x):
        _, last_states = self.convlstm(x)
        last = last_states[0][0]
        x = self.se1(F.relu(last))
        out = self.net(x)
        return torch.nn.functional.log_softmax(out, dim=1)


class Resnet18CNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # CNN definition
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        m = resnet18(pretrained=False)
        m.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = m.fc.in_features
        m.fc = nn.Linear(num_ftrs, 2)
        self.m = m

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.log_softmax(self.m(x), dim=1)


from torch.nn.utils import weight_norm

"""TCN adapted from https://github.com/locuslab/TCN"""


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class pad1d(nn.Module):
    def __init__(self, pad_size):
        super(pad1d, self).__init__()
        self.pad_size = pad_size

    def forward(self, x):
        return torch.cat([x, x[:, :, -self.pad_size:]], dim=2).contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalBlockTranspose(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 dropout=0.2):
        super(TemporalBlockTranspose, self).__init__()
        self.conv1 = weight_norm(nn.ConvTranspose1d(n_inputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding,
                                                    dilation=dilation))
        self.pad1 = pad1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.ConvTranspose1d(n_outputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding,
                                                    dilation=dilation))
        self.pad2 = pad1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.dropout1, self.relu1, self.pad1, self.conv1,
                                 self.dropout2, self.relu2, self.pad2, self.conv2)
        self.downsample = nn.ConvTranspose1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, hparams):
        super(TemporalConvNet, self).__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        time_len = 10

        num_inputs = input_dim
        num_channels = [5, 5, 5]
        kernel_size = 3
        dropout = 0.2
        layers = []
        layers_time = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = time_len if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers_time += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                          padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.network_time = nn.Sequential(*layers_time)

        hidden_size = num_channels[-1] * time_len + num_channels[-1] * input_dim
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.drop1 = torch.nn.Dropout(0.25)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.drop2 = torch.nn.Dropout(0.25)
        self.fc3 = torch.nn.Linear(hidden_size // 2, 2)
        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            self.drop1,
            self.relu,
            self.fc2,
            self.drop2,
            self.relu,
            self.fc3
        )

    def forward(self, x):
        y = self.network_time(x)
        x = x.permute([0, 2, 1])
        x = self.network(x)
        x = torch.cat([torch.flatten(x, 1), torch.flatten(y, 1)], 1)
        x = self.fc_nn(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma

    def forward(self, input_tensor, target_tensor):
        log_prob = input_tensor
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, log_preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(log_preds.device)

        n = log_preds.size(-1)
        log_preds = F.log_softmax(log_preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)
