import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


class NNUE(nn.Module):
    def __init__(self, stack_limit=10, accumulator_count=512):
        super(NNUE, self).__init__()
        self.accumulators = nn.Linear(64*(8+stack_limit*2), accumulator_count)
        self.output = nn.Linear(accumulator_count, 1)

    def forward(self, board_rep):
        accum = self.accumulators(board_rep)
        crelu = F.relu6(accum)
        out = self.output(crelu)
        tanh = F.tanh(out)
        return tanh


def test(net, dataset, batch_size=64):
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.eval()
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_count = 0
    loss_sum = 0.

    for idx, batch in enumerate(test_loader):
        s, target_p, target_v = batch
        s = s.float()
        if cuda:
            s, target_v = s.cuda(), target_v.cuda()
        predicted_v = net(s)

        loss_sum += F.mse_loss(predicted_v, target_v).item()

        test_count += 1

    return loss_sum / test_count


def train(net, dataset, class_weights, epochs, batch_size, optimizer):
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    else:
        print("WARNING: running on CPU, cuda not available!")
    net.train()

    train_set_size = int(len(dataset) * 0.95)

    validation_set_size = len(dataset) - train_set_size
    train_set, validation_set = data.random_split(dataset, [train_set_size, validation_set_size], generator=torch.Generator().manual_seed(42))

    train_weights = [class_weights[int(t[0])] for _, _, t in train_set]

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              sampler=WeightedRandomSampler(train_weights, num_samples=int(0.7*len(train_weights))))

    for epoch in range(epochs):
        loss_sum = 0.
        for idx, batch in tqdm(enumerate(train_loader), desc='training', total=1+int(0.7*len(train_weights)/batch_size)):
            s, target_p, target_v = batch
            s = s.float()
            target_v = target_v.float()

            if cuda:
                s, target_v = s.cuda(), target_v.cuda()
            optimizer.zero_grad()
            predicted_v = net(s)

            loss = F.mse_loss(predicted_v, target_v)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            if idx % 100 == 99:
                # print("completed batch " + str(idx+1) + "! current loss: " + str(loss_sum/100.))
                loss_sum = 0.

        validation_loss = test(net, validation_set, batch_size)
        return validation_loss
