import torch
import torch.nn.functional as F
import torch.utils.data as data

import pytak.ptn_parser as ptn_parser
from pytak.tak import GameState

from neural.model import TakNetwork, train, test

from dataset_builder import DatasetBuilder, get_input_repr, get_move_from_conv_repr

import numpy as np
import sys

files = [
    "../data/games_1.ptn",
    "../data/games_2.ptn",
    "../data/games_3.ptn",
    "../data/tiltak_tako_1.ptn",
    "../data/tiltak_tako_2.ptn",
    "../data/tiltak_tako_3.ptn",
    "../data/tiltak_tako_4.ptn",
]

test_file = "../data/games_test.ptn"

net = TakNetwork(stack_limit=10, res_blocks=10, filters=128)

lr = 0.01
for epoch in range(5):
    print("starting epoch " + str(epoch+1) + " for real!")
    for f in files:
        builder = DatasetBuilder(add_symmetries=True, ignore_plies=4)
        ptn_parser.main(f, builder)

        train(net, builder, epochs=1, batch_size=512, lr=lr)
    builder = DatasetBuilder(add_symmetries=True, ignore_plies=4)
    ptn_parser.main(test_file, builder)

    print("---TEST---")
    acc, top5_acc = test(net, builder, epochs=1, batch_size=512)
    print("acc: ", acc)
    print("top5 acc: ", top5_acc)

    lr = lr / 1.4

torch.save(net, 'model_10_128')
