from typing import Union
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from pytak.position_processor import PositionProcessor
from pytak.tak import GameState

offset_dict = {
    '1':      0,
    '11':     1,
    '111':    2,
    '1111':   3,
    '11111':  4,
    '11112':  5,
    '1112':   6,
    '11121':  7,
    '1113':   8,
    '112':    9,
    '1121':  10,
    '11211': 11,
    '1122':  12,
    '113':   13,
    '1131':  14,
    '114':   15,
    '12':    16,
    '121':   17,
    '1211':  18,
    '12111': 19,
    '1212':  20,
    '122':   21,
    '1221':  22,
    '123':   23,
    '13':    24,
    '131':   25,
    '1311':  26,
    '132':   27,
    '14':    28,
    '141':   29,
    '15':    30,
    '2':     31,
    '21':    32,
    '211':   33,
    '2111':  34,
    '21111': 35,
    '2112':  36,
    '212':   37,
    '2121':  38,
    '213':   39,
    '22':    40,
    '221':   41,
    '2211':  42,
    '222':   43,
    '23':    44,
    '231':   45,
    '24':    46,
    '3':     47,
    '31':    48,
    '311':   49,
    '3111':  50,
    '312':   51,
    '32':    52,
    '321':   53,
    '33':    54,
    '4':     55,
    '41':    56,
    '411':   57,
    '42':    58,
    '5':     59,
    '51':    60,
    '6':     61,
    }

class DatasetBuilder(PositionProcessor):

    def __init__(self):
        self.inputs   = [] # list of np arrays. contains board representation as input to network
        self.values   = [] # list of np arrays. contains target values for positions
        self.policies = [] # list of np arrays. contains target policies for positions

        self.result = 0.0  # float, target value for current game

    def add_game(self, size: int, playtak_id: int, white_name: str, black_name: str, ptn: str, result: str, rating_white: int, rating_black: int) -> int:
        if result[1] == '/':
            self.result = 0.0
        elif result[0] == '0':
            self.result = -1.0
        elif result[2] == '0':
            self.result == 1.0
        else:
            print("ERROR: can't parse game result!")

    def add_position(self, game_id: int, move, result: str, tps: str, next_tps: Union[str, None], tak: GameState):
        if move == None:
            return
        print("representation for tps " + tps + " with move " + move + ":")

        input = get_input_repr(tak)
        print("input: ", input)

        # create value np array
        # result is inverted, as board is always transformed to current player perspective
        value = np.array([self.result if tak.player == "white" else -self.result])
        print("value: ", value)

        policy = get_conv_move_repr(move)
        print("policy: ", policy)

        self.inputs.append(input)
        self.values.append(value)
        self.policies.append(policy)

# input representation: channels first. channels*height*width.
# channels: (w = current player, b = other player)
# - 6 for top stone: w_cap, b_cap, w_wall, b_wall, w_flat, b_flat
# - 2 for 7 remaining stones each: w_flat, b_flat (top to bottom)
def get_input_repr(board: GameState):
    input = np.zeros((20,6,6), dtype=float)

    for x in range(6):
        for y in range(6):
            stack = board.board[x][y].stones
            if len(stack) > 0:
                top_stone = stack[-1]
                if top_stone.stone_type == 'F':
                    idx = 0
                elif top_stone.stone_type == 'S':
                    idx = 2
                elif top_stone.stone_type == 'C':
                    idx = 4
                else:
                    print("ERROR: invalid stone type " + top_stone.stone_type)
                    continue
                if top_stone.colour != board.player:
                    idx = idx+1
                input[idx, x, y] = 1.0
                idx = 6
                for stone in reversed(stack[:-1]): # ignore top stone
                    if idx > 18:
                        break
                    use_idx = idx if stone.colour == board.player else idx+1
                    input[use_idx, x, y] = 1.0
                    idx = idx+2

    return input

# flat move representation like in chess. 4572 length np array
def get_flat_move_repr(move: str):
    # TODO
    policy = np.zeros((4572), dtype=float)
    return policy

# move representation as output from fully convolutional network. for each
# square, channels are:
# - 3 for each placement type
# - 62 for each spread direction, for each possible move permutation ignoring
#   board edge
#
# total of 9036 values, about half don't correspond to valid moves on board.
def get_conv_move_repr(move: str):
    policy = np.zeros((3+4*62,6,6), dtype=float)

    # check for move command:
    if any(x in move for x in ('>', '<', '+', '-')):
        offset = 3
        for x in ('>', '-', '<', '+'):
            if x in move:
                perm = move.split(x)[1]
                ptn = move.split(x)[0][1:]
                break
            offset = offset + 62
        idx = offset + offset_dict[perm]

    else:  # place command
        # check for special stones
        stone_type = 'F'
        ptn = move
        if move[0].isupper():
            stone_type = move[0]
            ptn = move[1:]

        # get target square
        if stone_type == 'F':
            idx = 0
        elif stone_type == 'S':
            idx = 1
        elif stone_type == 'C':
            idx = 2
        else:
            raise Exception("ERROR: invalid stone type " + stone_type)

    (x, y) = get_square(ptn)
    policy[idx,x,y] = 1.0
    return policy

def get_square(ptn: str):
    x = ord(ptn[0].lower()) - 97
    y = int(ptn[1]) - 1
    return x, y
