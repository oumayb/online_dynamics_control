import torch
from odc.trainer import DAETrainer
import argparse
import os
import torch.multiprocessing as mp
import json

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--batch_size_train', default=16, type=int)
parser.add_argument("--batch_size_valid", default=16, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--decay_lr_every', default=20, type=int)
parser.add_argument('--lr_decay', default=0.5, type=float)
parser.add_argument('--n_epochs', default=200, type=int)
parser.add_argument('--r', default=15, type=int)
parser.add_argument('--dt', default=0.1, type=float, help='1 / frame rate, DMD')
parser.add_argument('--b_learnt', default=0, type=int)
parser.add_argument('--a_method', default='prox', type=str, choices=['svd', 'pinv', 'still', 'qr', 'prox', 'lagrangian',
                                                                     'learnt'])

parser.add_argument('--online_update', default=0, type=int)
parser.add_argument('--measured_states', default=[], nargs='+',
                    help='Measured states after n_frames_A. Only considered if online_update is 1')
parser.add_argument('--longterm_pen', default=0, type=int)
parser.add_argument('--online_update_horizon', default=0, type=int,
                    help='for how much time steps do we want to do online update. '
                         'If 0, will use all states after n_frames_A to update. If 10, will only use 10')
parser.add_argument('--delayed_dmd', default=0, type=int)
parser.add_argument('--ddmdc', default=0, type=int, help='whether or not dataset with control')
parser.add_argument('--control_d', default=0, type=int, help='Dimension of control if ddmdc')
parser.add_argument('--history', default=1, type=int, help='History frames to predict from')
parser.add_argument('--n_frames_A', type=int, default=50,
                    help='Number of frames to use to build A matrix, if mdoel=DMDAE')
parser.add_argument('--normalize_control_inputs', default=0, type=int)
parser.add_argument('--bn_control', default=0, type=int)
parser.add_argument('--dtype', type=str, default='float')

parser.add_argument('--missing_states', type=int, default=[], nargs='+',
                   help='Whether or not to predict for a missing state. int entered will be the removed state')
parser.add_argument('--a_return', type=str, default='full', choices=['full', 'reduced'])
parser.add_argument('--m_prox', default=10, type=int,
                    help='Number of iterations for iterative refinement, if a_method is prox')
parser.add_argument('--lam_prox', default=1e6, type=float,
                    help='1/lam_prox is used for iterative refinement, if a_method is prox')
parser.add_argument('--reg', default=1e-8, type=float, help='Regularization coeff for A when a_method==lagrangian')

#parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--dist', default=0, type=int, help='Whether or not to multi-gpu')
parser.add_argument('--pin_memory', default=1, type=int)
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--save_weights', type=int, default=0)
parser.add_argument('--optim', default='adam', type=str)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--dampening', default=0.9, type=float)
parser.add_argument('--betas', default=[0.5, 0.999], type=list)
parser.add_argument('--weight_decay', default=1e-5, type=float)


parser.add_argument('--dataset_type', type=str, default='pendulum', choices=['pendulum', 'flow'])
parser.add_argument('--val_frac', default=0.1, type=float, help='Fraction of val data, only used from flow dataset')
parser.add_argument('--n_sequences_flow', type=int, default=1200)
parser.add_argument('--max_num_flow', type=int, default=5001)
parser.add_argument('--flow_channel', type=int, default=0)

parser.add_argument('--exp_name', type=str)
parser.add_argument('--data_path', type=str,
                    default="../../datasets/pinocchio_pendulum_resized_bw")
parser.add_argument('--sample_duration', default=100, type=int)
parser.add_argument('--video_fmt', default='video', type=str)
parser.add_argument('--dataset_fraction', default=1, type=float, help='Fraction of the dataset to use for train')
parser.add_argument('--dataset_fraction_test', default=1, type=float, help='Fraction of the dataset to use for val')
parser.add_argument('--model_name', type=str, default='DMDAE', choices=['DMDAE', 'DAE', 'CNNAE'])
parser.add_argument('--autoencoder_name', type=str, default="CNNAE",
                    help="Encoder to use with DAE model that encodes/decodes to videos.",
                    choices=['CNNAE'])
parser.add_argument('--vgg_layer', type=str, default='c4', choices=['c3', 'c4', 'c5'],
                    help='level of VGG features to use')
parser.add_argument('--n_channels', type=int, help='Number of channels in input video frames', default=1)
parser.add_argument('--ae_out_planes',type=int, default=[16, 32, 64, 64, 32, 8], nargs='+')
parser.add_argument('--strides', type=int,default=[1, 1, 1, 1, 1, 1], nargs='+')
parser.add_argument('--paddings', type=int,default=[1, 1, 1, 1, 1, 1], nargs='+')
parser.add_argument('--mlp_in_ae', type=int, default=0, help='Whether to have a MLP in the bottleneck of the AE')
parser.add_argument('--linear_neurons', type=int, default=[500, 100, 30], nargs='+')
parser.add_argument('--conv_o_dim', type=int, default=12800)

parser.add_argument('--ae_out_dim', type=int, default=30)
parser.add_argument('--reco_method', type=str, choices=['one_step', 'iter'], default='iter')
parser.add_argument('--ae_resume_path', type=str, default=None, help='path to model of AE to load from')

parser.add_argument('--overfit', default=0, type=int)

parser.add_argument('--loss', type=str, default="MSE", choices=['MSE'])
parser.add_argument('--loss_reduction', type=str, default='mean', choices=['sum', 'mean'])
parser.add_argument('--alpha_res', type=float, default=0, help='weight of residual loss')
parser.add_argument('--alpha_reco_first', type=int, default=1)
parser.add_argument('--alpha_reco_last', type=int, default=1)
parser.add_argument('--alpha_reco_missing', type=int, default=10)
parser.add_argument('--alpha_features_cycle', type=float, default=0)

parser.add_argument('--write_videos', default=4, type=int, help='whether or not to add videos to summarywriter')
parser.add_argument('--save_grads', default=0, type=int, help='whether or not to add gradients to summarywriter')
parser.add_argument('--log', default="epoch", type=str, choices=['epoch', 'iter'],
                    help='whether to log at iteration or epoch level')
parser.add_argument('--save', default=1, type=int, help='whether or not to create save_dir')



parser.add_argument('--resume_path', default=None, type=str)
parser.add_argument('--config', type=str, help='Path to config to start from, cancels all other arguments',
                    default=None)
args = parser.parse_args()

if args.config:
    with open(args.config, "r") as f:
        config = json.load(f)
else:
    config = vars(args)

config["exp_name"] = args.exp_name
config['config'] = args.config


args2 = dict()
args2["world_size"] = 1
args2["rank"] = 0


def init(gpu, ngpus_per_node=None, args=None):
    trainer = DAETrainer(config=config, gpu=gpu)
    trainer.train(gpu=gpu, ngpus_per_node=ngpus_per_node, args=args)

def main():
    if config["dist"]:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        ngpus_per_node = torch.cuda.device_count()
        args2["world_size"] = ngpus_per_node * args2["world_size"]
        mp.spawn(fn=init, nprocs=ngpus_per_node, args=(ngpus_per_node, args2), join=True)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        init(gpu=device)
        #init(gpu=args.device)
if __name__ == '__main__':
    main()
