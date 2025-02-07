# region Import

import yaml
import argparse

from easydict import EasyDict
from motion_vqvae import MoVQ

# endregion


# region Args Parser

parser = argparse.ArgumentParser(
    description="Pytorch implementation of Semantic Gesticulator"
)

parser.add_argument("--local-rank", default=0, type=int)

parser.add_argument('--config', type=str, default='')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train', action='store_true')
group.add_argument('--eval', action='store_true')

args = parser.parse_args()

# endregion


def main(args):
    print("start")
    with open(args.config, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    agent = MoVQ(config)
    print("into train")

    if args.train:
        agent.train()
    elif args.eval:
        agent.eval()
    else:
        raise ValueError


if __name__ == '__main__':
    main(args)