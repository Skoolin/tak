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

#net = torch.load('model_10_128', map_location=torch.device('cpu'))
#net.eval()

game = GameState(6)
example = torch.from_numpy(get_input_repr(game)).float()
print(example.size())

#traced_script_module = torch.jit.trace(net, example)
#traced_script_module.save("forward_10_128.pt")
