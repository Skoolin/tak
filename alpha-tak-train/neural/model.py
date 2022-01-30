import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MultiLoss(torch.nn.Module):
    def __init__(self, policy_weights=None):
        super(MultiLoss, self).__init__()
        self.policy_weights = torch.from_numpy(policy_weights).float() if policy_weights else None

    ### calculates combined loss of value and policy head.
    # target_p can be EITHER:
    # - index of the target move, where it would be in a tensor of same
    #   size and layout as predicted_p
    # - probability distribution tensor (sum 1.0) of moves with the same layout
    #   (flattened!) as predicted_p
    # predicted_v and target_v are single-float value tensors.
    # predicted_p has to be the flattened output of the policy head.
    def forward(self, predicted_p, target_p, predicted_v, target_v):
        value_loss = F.mse_loss(predicted_v, target_v)
        policy_loss = F.cross_entropy(predicted_p, target_p, weight=self.policy_weights)

        total_loss = (value_loss + policy_loss).mean()
        return total_loss

class InitialConvolution(nn.Module):
    def __init__(self, input_layers, filters):
        super(InitialConvolution, self).__init__()
        self.input_layers = input_layers
        self.filters = filters
        self.conv = nn.Conv2d(input_layers, filters, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(filters)

    def forward(self, s):
        s = s.view(-1, self.input_layers, 6, 6) # batch, channel, x, y
        s = self.conv(s)
        s = self.bn(s)
        s = F.relu(s)
        return s

class SE_Block(nn.Module):
    def __init__(self, filters, se_channels=32):
        super(SE_Block, self).__init__()
        self.globalAvgPool = nn.AvgPool2d(6, stride=1)
        self.fc1 = nn.Linear(filters, se_channels)
        self.flatten = nn.Flatten()
        self.w_fc = nn.Linear(se_channels, filters)
        self.b_fc = nn.Linear(se_channels, filters)

    def forward(self, s):
        r = self.globalAvgPool(s)
        r = self.flatten(r)
        r = F.relu(r)
        r = self.fc1(r)
        w = self.w_fc(r)
        w = w.view(s.size(0), s.size(1), 1, 1)
        w = F.sigmoid(w)
        b = self.b_fc(r)
        b = b.view(s.size(0), s.size(1), 1, 1)
        s = s*w.expand_as(s)+b.expand_as(s)
        return s

class ResBlock(nn.Module):
    def __init__(self, filters, se=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

        self.is_se = se
        if se:
            self.se = SE_Block(filters)

    def forward(self, s):
        residual = s
        s = self.conv1(s)
        s = self.bn1(s)
        s = F.relu(s)
        s = self.conv2(s)
        s = self.bn2(s)
        s += residual
        s = F.relu(s)

        if hasattr(self, "is_se") and self.is_se:
            s = self.se(s)
        return s

class PolicyHead(nn.Module):
    def __init__(self, filters):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, 3+4*62, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()

    def forward(self, s):
        s = self.conv1(s)
        s = self.bn1(s)
        s = F.relu(s)
        s = self.conv2(s)
        s = self.flatten(s)
        return s


class ValueHead(nn.Module):
    def __init__(self, filters):
        super(ValueHead, self).__init__()
        # having 32 output filters is supposed to yield better results!
        self.conv1 = nn.Conv2d(filters, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6*6*32, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, s):
        s = self.conv1(s)
        s = self.bn1(s)
        s = F.relu(s)
        s = self.flatten(s)
        s = self.fc1(s)
        s = F.relu(s)
        s = self.fc2(s)
        s = F.tanh(s)
        return s


class TakNetwork(nn.Module):
    def __init__(self, stack_limit=12, res_blocks=20, filters=256):
        super(TakNetwork, self).__init__()
        self.initial_conv = InitialConvolution(6+2*stack_limit+2+2*30, filters)
        self.res_blocks = nn.Sequential(*[ResBlock(filters) for x in range(res_blocks)])
        self.policy_head = PolicyHead(filters)
        self.value_head = ValueHead(filters)

    def forward(self, s):
        s = self.initial_conv(s)
        s = self.res_blocks(s)
        p = self.policy_head(s)
        v = self.value_head(s)
        return p, v

def test(net, dataset, batch_size=64):
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.eval()
    num_entries = len(dataset)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    top1_count = 0
    top5_count = 0

    for idx, batch in enumerate(test_loader):
        s, target_p, target_v = batch
        s = s.float()
        if cuda:
            s = s.cuda()
        predicted_p, predicted_v = net(s)

        pred_p_result = predicted_p.argmax(dim=1)
        pred_p_top5 = torch.topk(predicted_p, 5, dim=1)

        for i in range(len(target_p)):
            if pred_p_result[i] == target_p[i]:
                top1_count = top1_count+1
            i_pred_p_top5 = [pred_p_top5.indices[i][x] for x in range(5)]
            if target_p[i].item() in i_pred_p_top5:
                top5_count = top5_count+1

    return top1_count/num_entries, top5_count/num_entries

def train(net, dataset, epochs, batch_size, optimizer, policy_weights=None):
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.train()

    criterion = MultiLoss(policy_weights)

    train_set_size = int(len(dataset) * 0.95)
    validation_set_size = len(dataset) - train_set_size
    train_set, validation_set = data.random_split(dataset, [train_set_size, validation_set_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        loss_sum = 0.
        for idx, batch in enumerate(train_loader):
            s, target_p, target_v = batch
            s = s.float()
            target_v = target_v.float()
            if cuda:
                s, target_p, target_v = s.cuda(), target_p.cuda(), target_v.cuda()
            optimizer.zero_grad()
            predicted_p, predicted_v = net(s)

            loss = criterion(predicted_p, target_p, predicted_v, target_v)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if idx % 100 == 99:
                print("completed batch " + str(idx+1) + "! current loss: " + str(loss_sum/100.))
                loss_sum = 0.

        acc, top5_acc = test(net, validation_set, batch_size)
        return acc, top5_acc
