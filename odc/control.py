from odc.models import CNNAE, DMDAE
import torch
import argparse
import os
import matplotlib.pyplot as plt

from odc.pendulum_dataset import PendulumDataset
import json

from cvxpy import quad_form, norm, Problem, Minimize, Variable, Parameter, OSQP
import numpy as np

from odc.train_utils import plot_for, save_gif

import math


parser = argparse.ArgumentParser(description='Generate controls')
parser.add_argument('--exp_path', default='models/resume_cartpole_control_longterm_mes=180_190_2021_5_27_20_59',
                    type=str, help='Path to exp folder')
parser.add_argument('--model_path',
                    default='models/resume_cartpole_control_longterm_mes=180_190_2021_5_27_20_59/model_loss_0.0005677287699654698.pth',
                    type=str, help='Path to model ckpt to use')
parser.add_argument('--data_path', default='datasets/cartpole_control_64_64_bw', type=str, help='Path to data path')
parser.add_argument('--video_idx', default=1528, type=int, help='Idx of the video of the system to control')
parser.add_argument('--idx_init', default=99, type=int, help='Initial position')
parser.add_argument('--idx_final', default=61, type=int, help='Target poisition')
parser.add_argument('--results_dir', default='results', type=str, help='Where to save results')
parser.add_argument('--system', default='cartpole', type=str, help='Name of the system',
                    choices=['pendulum', 'cartpole'])
args = parser.parse_args()

exp_path = args.exp_path
config_path = os.path.join(exp_path, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# To run locally, set gpu to, data_path to local data_path, bs to a small one, num_workers to 0
config["gpu"] = 0
config['data_path'] = args.data_path
config['dtype'] = 'double'
config["save"] = 0
config["num_workers"] = 0

print('Loading dataset and model')

val_dataset = PendulumDataset(root_path=config["data_path"], split="test",
                              overfit=config["overfit"], sample_duration=config["sample_duration"],
                              video_fmt=config['video_fmt'], control=config['ddmdc'],
                              normalize=config['normalize_control_inputs'], dtype=config['dtype'])


model_path = args.model_path
if not config['gpu']:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
else:
    state_dict = torch.load(model_path)

state_dict_copy = dict()
for key in state_dict.keys():
    if args.system == 'cartpole':
        state_dict_copy[key[7:]] = state_dict[key]
    elif args.system == 'pendulum':
        state_dict_copy[key] = state_dict[key]

autoenc = CNNAE(config=config)
model = DMDAE(autoencoder=autoenc, r=config['r'], n_points_A=config['n_frames_A'], config=config)

model.load_state_dict(state_dict_copy)

model.eval()


idx = args.video_idx
idx_init = args.idx_init
idx_final = args.idx_final

batch = dict()
batch['data'] = val_dataset[idx]['data'].view((1, 200, 1, 64, 64))
batch['control_inputs'] = val_dataset[idx]['control_inputs'].view((1, 200, config['control_d']))
batch['name'] = val_dataset[idx]['name']
out = model(batch)
features = out['features'][0]
A_matrix = out['A_matrix'][0]
C = A_matrix[:, :config['ae_out_dim']].detach().numpy()
D = A_matrix[:, config['ae_out_dim']:-config['control_d']].detach().numpy()
B = A_matrix[:, -config['control_d']:].detach().numpy()
f0 = features[idx_init, :].detach().numpy()
f1 = features[idx_init +1, :].detach().numpy()
f_goal = features[idx_final, :].detach().numpy()

T = 0.5
dt_mes = 5e-2
dt_c = dt_mes
ratio = dt_mes / dt_c

Cc = ratio * C
Dc = ratio * D
Bc = ratio * B

n = config['ae_out_dim']
d = config['control_d']

N = math.floor(T / dt_c)

u_max = 5
R = 1e-5

# Define variables
x = Variable((n, N + 2))
u = Variable((d, N))
x_init_0 = Parameter(n)
x_init_0.value = f0
x_init_1 = Parameter(n)
x_init_1.value = f1


# Define costs for states and inputs
Q = 1 * np.eye(n)
QN = Q
R = R * np.eye(d)

# Construct optimization problem
cost = 0
constr = []
for t in range(1, N - 1):
    cost += quad_form((x[:, t + 2] - f_goal), Q) + quad_form(u[:, t], R)
    constr += [x[:, t + 2] == Cc @ x[:, t] + Dc @ x[:, t + 1] + (Bc @ u[:, t])]
    constr += [norm(u[:, t], 2) <= u_max]
# Terminal cost and terminal constraints
cost += quad_form((x[:, N + 1] - f_goal), QN) + quad_form(u[:, N - 1], R)
constr += [x[:, N + 1] == Cc @ x[:, N - 1] + Dc @ x[:, N] + (Bc @ u[:, N - 1])]

# Constraint on last position
constr += [x[:, N + 1] == f_goal]

# initial constraints
constr += [x[:, 0] == x_init_0]
constr += [x[:, 1] == x_init_1]

x0 = f0
x1 = f1

# Solve problem
print('Solving QP')
prob = Problem(Minimize(cost), constr)

# Update
n_updates = 20
for i in range(0, n_updates):
    x_init_0.value = x0
    x_init_1.value = x1
    prob.solve(warm_start=True)
    x_temp = x0
    x0 = x1
    x1 = Cc @ x1 + Dc @ x1 + Bc @ u.value[:, 0]
    if i == 0:
        x_lqr = x.value
        u_lqr = u.value

x_mpc = x.value
u_mpc = u.value


# Vizualise
advanced_features = x_lqr


features_dict = dict()
features_dict['latent'] = torch.tensor(advanced_features.reshape((1, 8, N+2))).transpose(1, 2).double()

x_reco = model.decode(features_dict)
reco = dict()
reco['x'] = x_reco

print('Saving img and gif')
# Save sequence of frames as png
plot_for(x_reco.transpose(1, 3).detach(), plot_func=plt.imshow, cmap='Greys_r', n_cols=6, figsize=(20, 5))
img_target_path = os.path.join(args.results_dir, '{}.png'.format(args.system))
plt.savefig(img_target_path, dpi=300, bbox_inches='tight')

# Save sequence of frames as gif
gif_target_path = os.path.join(args.results_dir, '{}.gif'.format(args.system))
save_gif(x_reco.transpose(1, 3)[:, :, :, 0].detach().numpy() * 255, target_path=gif_target_path, fps=10)