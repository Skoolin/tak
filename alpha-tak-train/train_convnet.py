import torch
import torch.optim as optim

import pytak.ptn_parser as ptn_parser

from neural.model import TakNetwork, train, test

from dataset_builder import DatasetBuilder

from ptflops import get_model_complexity_info

files = ["../data/train/games0_6s_train_"+str(i+1)+".ptn" for i in range(42)]
test_files = ["../data/test/games0_6s_test_"+str(i+1)+".ptn" for i in range(4)]


net = TakNetwork(stack_limit=15, res_blocks=16, filters=256)

macs, params = get_model_complexity_info(net, (6+2*15+2+2*30, 6, 6), as_strings=True,
                                         print_per_layer_stat=False, verbose=False)
print(f"Params: {params}")
print(f"MACs: {macs}")

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

        acc, top5_acc = train(net, builder, epochs=1, batch_size=1024, optimizer=optimizer)
        print("---validation---")
        print("acc: ", acc)
        print("top5 acc: ", top5_acc)
        scheduler.step(acc+0.3*top5_acc)  # if we stop improving, reduce LR!

    builder = DatasetBuilder(add_symmetries=False, ignore_plies=6)
    for f in test_files:
        ptn_parser.main(f, builder)

    print("---TEST---")
    acc, top5_acc = test(net, builder, batch_size=1024)
    print("acc: ", acc)
    print("top5 acc: ", top5_acc)

    # save current version of net
    torch.save(net, 'senet_08_08_2025_0004_large')
