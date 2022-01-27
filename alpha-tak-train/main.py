import torch
import torch.nn.functional as F
import torch.utils.data as data

import pytak.ptn_parser as ptn_parser
from pytak.tak import GameState

from neural.model import TakNetwork, train, test

from dataset_builder import DatasetBuilder, get_input_repr, get_move_from_conv_repr

import numpy as np
import sys

def predict_move(net, s):
    print("predicting move:")
    s = torch.from_numpy(s)
    net.eval()
    p, v = net(s.float())
    p = F.softmax(p, dim=1)
    top5 = torch.topk(p, 5, dim=1)
    for i in range(5):
        print(str(i+1) + ". " + get_move_from_conv_repr(top5.indices[0][i].item()) + ": " + str(top5.values[0][i].item()))


builder = DatasetBuilder(add_symmetries=True, ignore_plies=4)
ptn_parser.main("../data/games.ptn", builder)
np.set_printoptions(threshold=sys.maxsize)
print(len(builder))
# policy_weights = (500.+len(builder)/len(builder.policy_counts)) / builder.policy_counts

train_set_size = int(len(builder) * 0.9)
test_set_size = len(builder) - train_set_size
train_set, test_set = data.random_split(builder, [train_set_size, test_set_size], generator=torch.Generator().manual_seed(42))

net = TakNetwork(stack_limit=10, res_blocks=5, filters=48)
train(net, train_set, epochs=10, batch_size=128, lr=0.01)

torch.save(net, 'model_8_128')
#net = torch.load('model_8_128')

test(net, test_set, batch_size=128)

#exit()

# let it analyze one recent human game
builder = DatasetBuilder()
ptn_parser.main("../data/single.ptn", builder)
test(net, builder, batch_size=20)
