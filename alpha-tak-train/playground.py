import torch
import torch.nn.functional as F
import torch.utils.data as data

import pytak.ptn_parser as ptn_parser
from pytak.tak import GameState

from neural.model import TakNetwork, train, test

from dataset_builder import DatasetBuilder, get_input_repr, get_move_from_conv_repr

import numpy as np
import sys

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

builder = DatasetBuilder(add_symmetries=False, ignore_plies=4)
ptn_parser.main("../data/single.ptn", builder)

test(net, builder)

builder = DatasetBuilder(add_symmetries=False, ignore_plies=0)
ptn_parser.main("../data/single.ptn", builder)

game = GameState(6)

for i in range(4):
    move = get_move_from_conv_repr(builder.policies[i])
    print(move)
    game.move(move)

for i in range(4, 40):
    game.print_state()
    move = get_move_from_conv_repr(builder.policies[i])
    print("human move: " + move)
    predict_move(net, game)
    game.move(move)
