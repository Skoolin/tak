import torch
import torch.nn.functional as F
import torch.utils.data as data

import pytak.ptn_parser as ptn_parser
from pytak.tak import GameState

from neural.model import TakNetwork, train, test

from dataset_builder import DatasetBuilder, get_input_repr, get_move_from_conv_repr

import numpy as np
import matplotlib.pyplot as plt
import sys

# sys.stdout = open('test.txt', 'w')

def predict_move(net, game):
    print("predicting move:")
    s = get_input_repr(game)
    s = torch.from_numpy(s)
    net.eval()
    p, v = net(s.float())
    p = F.softmax(p, dim=1)
    top5 = torch.topk(p, 5, dim=1)
    for i in range(5):
        print(str(i+1) + ". " + get_move_from_conv_repr(top5.indices[0][i].item()) + ": " + str(top5.values[0][i].item()))

net = torch.load('model_8_128', map_location=torch.device('cpu'))

accs = []
top5_accs = []

for plie in range(4, 100, 2):
    print("testing depth ", int(2+plie/2))
    builder = DatasetBuilder(add_symmetries=False, ignore_plies=plie, max_plies=2)
    ptn_parser.main("../data/games_test.ptn", builder)
    acc, top5_acc = test(net, builder) if len(builder) > 0 else (0., 0.)
    accs.append(acc)
    top5_accs.append(top5_acc)

plt.title("accuracy over depth")
plt.xlabel("depth in moves")
plt.ylabel("accuracy")
plt.plot(np.arange(2, 50), accs, label="accuracy")
plt.plot(np.arange(2, 50), top5_accs, label="top5 accuracy")
plt.legend()
plt.show()
