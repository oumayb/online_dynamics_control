import json
import os

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default=None)

args = parser.parse_args()
arguments = vars(args)

config = dict()

for filename in os.listdir(args.data_dir):
    if 'u_' in filename:
        id = filename.split("_")[-1].split(".")[0]
        config_path = os.path.join(args.data_dir, filename)
        with open(config_path) as f:
            current_config = json.load(f)
        config[id] = current_config



target_path = os.path.join(args.data_dir, 'u.json')
with open(target_path, 'w') as f:
    json.dump(config, f)