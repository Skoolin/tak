import sys
import re

from tqdm import tqdm

from pytak.position_processor import PositionProcessor
from pytak.tak import GameState


def add_ptn(ptn, dp: PositionProcessor, max_plies=sys.maxsize):
    lines = ptn.split('\n')

    headers = {}
    all_moves = []
    result = '0-0'

    for line in lines:
        if len(line) < 4:
            continue
        if line.startswith('['):
            line = line[1:-1] # remove brackets
            split = line.split(' ')
            header_name = split[0]
            header_value = ' '.join(split[1:])
            header_value = header_value[1:-1] # remove "
            headers[header_name] = header_value
            if header_name == 'Result':
                result = header_value
            continue

        # remove comments
        line = re.sub(r'\{[^{}]*\}', '', line)

        words = line.split(' ')
        for word in words:
            if len(word) < 2:
                continue
            if word[-1] == '.': # move number
                continue
            if word == result: # game result on end (not required)
                continue

            # extend spreads to full size
            if any(x in word for x in ['-','+','<','>']):
                if not word[0].isnumeric():
                    word = '1'+word
                if not word[-1].isnumeric():
                    word = word+word[0]

            all_moves.append(word)

    # apply upper bound of ply depth
    all_moves = all_moves[0:min(len(all_moves), max_plies)]

    size = int(headers["Size"])
    playtak_id = headers["playtak_id"] if "playtak_id" in headers.keys() else 0
    white_name = headers["Player1"]
    black_name = headers["Player2"]
    rating_white = headers["Rating1"] if "Rating1" in headers.keys() else 1000
    rating_black = headers["Rating2"] if "Rating2" in headers.keys() else 1000

    # create board
    tak = GameState(6)

    # add game to database
    game_id = dp.add_game(size, playtak_id, white_name, black_name, ptn, result, rating_white, rating_black)

    # make all moves
    for i in range(0, len(all_moves)):
        last_tps = tak.get_tps()
        last_move = all_moves[i]
        dp.add_position(game_id, last_move, result, last_tps, tak.get_tps(), tak)
        tak.move(all_moves[i])
    dp.add_position(game_id, None, result, tak.get_tps(), None, tak)


def main(ptn_file, dp: PositionProcessor):

    max_plies = 400

    f = open(ptn_file)
    count = f.read().count('\n\n[')+1
    f.close()

    with tqdm(total=count, mininterval=10.0, maxinterval=50.0) as progress:
        with open(ptn_file) as f:
            ptn = ''
            line = f.readline()
            parse_headers = True
            while line:
                if line.startswith("["):
                    if not parse_headers:
                        add_ptn(ptn, dp, max_plies)
                        parse_headers = True
                        progress.update()
                        ptn = ''
                else:
                    parse_headers = False
                ptn += line
                line = f.readline()
            add_ptn(ptn, dp, max_plies)
            progress.update()
