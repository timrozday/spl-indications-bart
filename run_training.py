from utils import *
import yaml
import argparse

import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Train BART model')
parser.add_argument('-a', '--args_file', type=str, default='./args.yaml',
                    help='Path to YAML config file containing model and training parameters.')
parser.add_argument('-e', '--extra_args_file', type=str,
                    help='Path to YAML config file containing extra model and training parameters.')

cli_args = parser.parse_args()

with open(cli_args.args_file) as f:
    args_dict = yaml.load(f)

if cli_args.extra_args_file:
    with open(cli_args.extra_args_file) as f:
        extra_args_dict = yaml.load(f)
        for k,v in extra_args_dict.items():
            args_dict[k] = v

args = Dict2Obj(args_dict)

ner_system = RobertaSystem(args)  # Load model
trainer = get_trainer(ner_system, args)  # get the trainer

trainer.fit(ner_system)  # Train!
