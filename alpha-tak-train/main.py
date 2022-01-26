import pytak.ptn_parser as ptn_parser
from dataset_builder import DatasetBuilder

builder = DatasetBuilder()
ptn_parser.main("../data/games.ptn", builder)
