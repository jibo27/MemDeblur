import yaml
from types import SimpleNamespace
import argparse
import os

from train import Trainer

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = ParseArgs()
    with open(args.config, mode='r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = SimpleNamespace(**config)
    config.exp_name = os.path.basename(args.config)[:-4]

    trainer = Trainer(config, config_path=args.config)
    trainer.run()
