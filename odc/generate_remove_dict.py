from tqdm import tqdm

import os
from shutil import rmtree
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--runs_dir_path", type=str, help="Directory that contains runs dir")
parser.add_argument('--remove_dict_path', type=str, default="/gpfswork/rech/vgn/ubc18vn/pendulum/remove_dict.json")
args = parser.parse_args()
to_remove = []  # to_remove will be a list of exp names that should be removed

runs_dir = args.runs_dir_path

remove_dict = dict()
remove_dict["exp_names"] = list(set(to_remove))




def get_losses_to_remove(runs_dir, loss_type='model_loss'):
    remove_loss_paths = []
    for exp_name in tqdm(os.listdir(runs_dir)):
        if '.DS' not in exp_name:
            losses = []
            exp_path = os.path.join(runs_dir, exp_name)
            for file in os.listdir(exp_path):
                if loss_type in file:
                    loss = float(file.split('.pth')[0].split("_")[-1])
                    losses.append(loss)
                else:
                    continue
            if len(losses) > 0:
                min_loss = np.min(losses)
                for file in os.listdir(exp_path):
                    if loss_type in file:
                        if str(min_loss) not in file:
                            path = os.path.join(exp_name, file)
                            remove_loss_paths.append(path)
    return remove_loss_paths


remove_loss_paths_reco = get_losses_to_remove(runs_dir=runs_dir, loss_type='reco_loss')
remove_loss_paths_res = get_losses_to_remove(runs_dir=runs_dir, loss_type='residual_loss')
remove_loss_paths_last_reco = get_losses_to_remove(runs_dir=runs_dir, loss_type='reco_last_loss')
remove_loss_paths_first_reco = get_losses_to_remove(runs_dir=runs_dir, loss_type='reco_first_loss')
remove_loss_paths_loss = get_losses_to_remove(runs_dir=runs_dir, loss_type='model_loss')

remove_loss_paths = remove_loss_paths_reco + remove_loss_paths_res + remove_loss_paths_last_reco + remove_loss_paths_first_reco + remove_loss_paths_loss

remove_dict["loss_paths"] = list(set(remove_loss_paths))

with open(args.remove_dict_path, "w") as f:
    json.dump(remove_dict, f)