import os
import argparse

from dataloader import create_subdataset

argparser = argparse.ArgumentParser()

argparser.add_argument("--imgs")
argparser.add_argument("--labels")
argparser.add_argument("--output")
argparser.add_argument("--valsize", type=int, default=10) # trainsize = valsize * 5

args = argparser.parse_args()


create_subdataset(args.imgs, args.labels, args.output, args.valsize)
