import torch
import torch.optim as optim

import pytak.ptn_parser as ptn_parser

from neural.nnue import NNUE, train, test

from dataset_builder import DatasetBuilder

files = ["../data/train/games0_6s_train_"+str(i+1)+".ptn" for i in range(42)]
test_files = ["../data/test/games0_6s_test_"+str(i+1)+".ptn" for i in range(4)]

considered_captives = 10

net = NNUE(considered_captives, 512)

lr = 0.03
optimizer = optim.Adam(net.parameters(), lr=lr)
print("---training---")

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, cooldown=20)
for i in range(5):
    print("---- EPOCH ", i+1, " ----")
    for f in files:
        # print("loading training file: ", f)
        builder = DatasetBuilder(add_symmetries=True, ignore_plies=6, nnue=True,
                                 considered_captives=considered_captives)
        ptn_parser.main(f, builder)
        val_loss = train(net, builder, epochs=1, batch_size=512, optimizer=optimizer)
        scheduler.step(val_loss)
        print("validation loss: ", val_loss)

    builder = DatasetBuilder(add_symmetries=False, ignore_plies=6, nnue=True, considered_captives=considered_captives)
    for f in test_files:
        ptn_parser.main(f, builder)

    print("---TEST---")
    test_loss = test(net, builder, batch_size=512)
    print("test loss: ", test_loss)

    # save current version of net
    torch.save(net, 'nnue_09_08_2025_0001')
