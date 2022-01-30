import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import pytak.ptn_parser as ptn_parser
from pytak.tak import GameState

from neural.model import TakNetwork, train, test

from dataset_builder import DatasetBuilder, get_input_repr, get_move_from_conv_repr

import numpy as np
import sys

files = ["../data/train/games0_6s_train_"+str(i+1)+".ptn" for i in range(42)]
test_files = ["../data/test/games0_6s_test_"+str(i+1)+".ptn" for i in range(4)]


net = TakNetwork(stack_limit=15, res_blocks=10, filters=128)

lr = 0.01
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=6, cooldown=2)
for epoch in range(4):
    print("starting epoch ", epoch+1)
    for f in files:
        print("---training---")
        print("training file: ", f)
        builder = DatasetBuilder(add_symmetries=True, ignore_plies=6)
        ptn_parser.main(f, builder)

        acc, top5_acc = train(net, builder, epochs=1, batch_size=512, optimizer=optimizer)
        print("---validation---")
        print("acc: ", acc)
        print("top5 acc: ", top5_acc)
        scheduler.step(acc+0.3*top5_acc) # if we stop improving, reduce LR!

    builder = DatasetBuilder(add_symmetries=False, ignore_plies=6)
    for f in test_files:
        ptn_parser.main(f, builder)

    print("---TEST---")
    acc, top5_acc = test(net, builder, batch_size=512)
    print("acc: ", acc)
    print("top5 acc: ", top5_acc)

    # save current version of net
    torch.save(net, 'model_10_128')
