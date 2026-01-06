import random

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import pytak.ptn_parser as ptn_parser

from neural.nnue import NNUE, train, test

from dataset_builder import DatasetBuilder

files = ["../data/train/games0_6s_train_"+str(i+1)+".ptn" for i in range(42)]
test_files = ["../data/test/games0_6s_test_"+str(i+1)+".ptn" for i in range(4)]

considered_captives = 10

# calculate class weights
class_counts = [0, 0, 0]

for f in tqdm(files, desc="balancing dataset"):
    builder = DatasetBuilder(add_symmetries=True, ignore_plies=6, nnue=True, considered_captives=considered_captives)
    ptn_parser.main(f, builder)
    targets = torch.tensor(np.array([builder[i][2] for i in range(len(builder))]))
    targets_np = targets.numpy()
    unique, counts = np.unique(targets_np, return_counts=True)
    for i in range(len(unique)):
        class_counts[int(unique[i])] += counts[i]

class_weights = [sum(class_counts) / (class_counts[i] + 1.0) for i in range(len(class_counts))]
class_weights [0] *= 0.25  # draws are less useful!
class_weights = np.array(class_weights)

net = NNUE(considered_captives, 512)

lr = 0.0001
optimizer = optim.Adam(net.parameters(), lr=lr)
print("---- training ----")

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for i in range(30):
    print("---- EPOCH ", i+1, " ----")
    random.seed(100+i*17)
    random.shuffle(files)
    builder = DatasetBuilder(add_symmetries=True, ignore_plies=6, nnue=True,
                             considered_captives=considered_captives, seed=100 + i*17)
    for f in tqdm(files, desc="loading training data"):
        ptn_parser.main(f, builder)

    val_loss = train(net, builder, class_weights, epochs=1, batch_size=512, optimizer=optimizer)

    builder = DatasetBuilder(add_symmetries=True, ignore_plies=6, nnue=True, considered_captives=considered_captives, seed=42)
    for f in test_files:
        ptn_parser.main(f, builder)

    test_loss = test(net, builder, batch_size=512)

    scheduler.step()
    print("test loss: ", test_loss)

    # save current version of net
    torch.save(net, 'nnue_06_01_2026_0001')
