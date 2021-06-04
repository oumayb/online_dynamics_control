import os
import shutil
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--runs_dir_path", type=str, help="Directory that contains runs dir")
parser.add_argument("--dict_path", type=str, help="Path to dict that contains paths to remove")
parser.add_argument('--delete', type=int, default=0, help='whether to delete first ckpts or keep them')
args = parser.parse_args()


with open(args.dict_path, "r") as f:
    remove_dict = json.load(f)

exp_names = remove_dict["exp_names"]
loss_paths = remove_dict["loss_paths"]


for exp_name in exp_names:
    exp_path = os.path.join(args.runs_dir_path, exp_name)
    if os.path.exists(exp_path):
        shutil.rmtree(exp_path)


for loss_path in tqdm(loss_paths):
    loss_path_remove = os.path.join(args.runs_dir_path, loss_path)
    if os.path.exists(loss_path_remove):
        os.remove(loss_path_remove)
