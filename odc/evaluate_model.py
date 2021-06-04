import torch
from odc.trainer import DAETrainer
import argparse
import os
import torch.multiprocessing as mp
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Evaluate all experiments in a given dir: runs_dir')
parser.add_argument('--batch_size_train', default=8, type=int)

parser.add_argument('--dist', default=0, type=int, help='Whether or not to multi-gpu')

parser.add_argument('--runs_dir', type=str, default=None, help='path to folder with experiments folders')
parser.add_argument('--sample_duration', type=int, default=None)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--measured_states', default=[], nargs='+', help='Measured states after n_frames_A')
parser.add_argument('--online_update', type=int, default=0)
parser.add_argument('--metric_prefix', type=str, default='')
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--data_path', type=str, default=None)
args = parser.parse_args()

args2 = dict()
args2["world_size"] = 1
args2["rank"] = 0


def init(gpu, config, ngpus_per_node=None, args=None):
    trainer = DAETrainer(config=config, gpu=gpu)
    trainer.evaluate(gpu=gpu, ngpus_per_node=ngpus_per_node, args=args)


def main():
    for i, exp in tqdm(enumerate(os.listdir(args.runs_dir))):
        if args.exp_name:
            if args.exp_name not in exp:
                continue
        resume_path = None
        exp_path = os.path.join(args.runs_dir, exp)
        config_path = os.path.join(exp_path, 'config.json')
        with open(config_path, "r") as f:
            config = json.load(f)
        if 'mse.npy' in os.listdir(exp_path) and args.metric_prefix == '':
            continue
        # Override config arguments that were input to the parser
        if args.sample_duration:
            config['sample_duration'] = args.sample_duration
        if args.measured_states:
            config['measured_states'] = args.measured_states
        if args.metric_prefix:
            config['metric_prefix'] = args.metric_prefix
        config['data_path'] = args.data_path
        config['batch_size_train'] = args.batch_size_train
        config['dist'] = args.dist
        config['save'] = args.save
        config['exp_path'] = exp_path
        config['eval'] = args.eval
        if args.online_update:
            config['online_update'] = args.online_update
        for filename in os.listdir(exp_path):
            if 'reco_last_loss' in filename:  # We use the model with the best reco loss over the predicted frames
                resume_path = os.path.join(exp_path, filename)
                continue  # gets out from for loop
        # If there was only one loss that was saved:
        if resume_path:
            pass
        else:
            for filename in os.listdir(exp_path):
                if 'model_loss' in filename:
                    resume_path = os.path.join(exp_path, filename)
                    continue  # gets out from for loop
        try: resume_path
        except:
            resume_path = None
        config['resume_path'] = resume_path
        if config['resume_path'] == None:
            print("No model")
            continue  # gets out from for loop (changes exp folder)
        if config["dist"]:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            ngpus_per_node = torch.cuda.device_count()
            args2["world_size"] = ngpus_per_node * args2["world_size"]
            mp.spawn(fn=init, nprocs=ngpus_per_node, args=(config, ngpus_per_node, args2), join=True)
        else:
            init(gpu=0, config=config)


if __name__ == '__main__':
    main()
