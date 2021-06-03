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
    exp_name = loss_path.split('model')[0]
    exp_path = os.path.join(args.runs_dir_path, exp_name)
    if not args.delete:
        all_ckpts_path = os.path.join(exp_path, 'all_checkpoints')
        if not os.path.isdir(all_ckpts_path):
            os.mkdir(all_ckpts_path)
    exp_name = loss_path.split('/')[0]
    if os.path.exists(loss_path):
        os.remove(loss_path)