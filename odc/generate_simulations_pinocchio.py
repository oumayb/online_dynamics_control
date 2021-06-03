import argparse
from tqdm import tqdm

import numpy as np

import os

import random

import json

import pinocchio as pin

from odc.pinocchio_pendulum_utils import create_model, generate_pendulum_pinocchio

from multiprocessing import Process

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--idx_start', type=int, default=None)
parser.add_argument('--idx_end', type=int, default=None)

parser.add_argument('--min_bar_length', type=float, default=0.3)
parser.add_argument('--max_bar_length', type=float, default=0.8)
parser.add_argument('--min_bar_ratio', type=float, default=0.8)
parser.add_argument('--max_bar_ratio', type=float, default=1.2)
parser.add_argument('--body_mass', type=float, default=1., help='Mass of the first mass')
parser.add_argument('--min_mass_ratio', type=float, default=0.5)
parser.add_argument('--max_mass_ratio', type=float, default=2)
parser.add_argument('--lower_limit', default=2, type=int)
parser.add_argument('--upper_limit', default=3, type=int)
parser.add_argument('--min_initial_v', default=0, type=float)
parser.add_argument('--max_initial_v', default=0, type=float)
parser.add_argument('--scene_scale', type=float, default=0.8)
parser.add_argument('--window_size', type=int, default=128)
parser.add_argument('--fps', type=int, default=20)
parser.add_argument('--dt', type=float, default=0.005)
parser.add_argument('--damping', type=float, default=0)
parser.add_argument('--duration', type=float, default=5, help='duration in seconds')
parser.add_argument('--target_dir', type=str)
parser.add_argument('--N', type=int, default=1, help='Number of pendulums')
parser.add_argument('--system_position', type=float, default=0.35, help='Position of the system')
parser.add_argument('--with_cart', default=0, type=int)
parser.add_argument('--cart_mass', default=5., type=float)
parser.add_argument('--min_cart_mass', default=1, type=float)
parser.add_argument('--max_cart_mass', default=10, type=float)
parser.add_argument('--min_cart_length_radius_ratio', type=float, default=3)
parser.add_argument('--max_cart_length_radius_ratio', type=float, default=7)
parser.add_argument('--cart_radius', default=0.1, type=float)

parser.add_argument('--ref_traj', default=0, type=int)
parser.add_argument('--Kv', default=0, type=int)
parser.add_argument('--Kp', default=0, type=int)
parser.add_argument('--f_ref', type=float, default=0.3, help='Max frequency range for target sinusoid, in rad/s')
parser.add_argument('--a_ref', type=float, default=1, help='Max amplitude range for target sinusoid. Fraction of 2 pi rad')

parser.add_argument('--save_qs', default=0, type=int)


args = parser.parse_args()
arguments = vars(args)

with open(os.path.join(args.target_dir, 'arguments.json'), 'w') as f:
    json.dump(arguments, f)

config = dict()

def generate_sample(i, q, v):
    save_path = os.path.join(args.target_dir, '{}.mp4'.format(i[0]))
    if args.ref_traj:
        controls_path = os.path.join(args.target_dir, 'u_{}.json'.format(i[0]))
    controls, qs = generate_pendulum_pinocchio(save_path=save_path, q=q, v=v, geom_model=geom_model,
                                               visual_model=visual_model, model=model, scene_scale=args.scene_scale,
                                               window_size=args.window_size, dt=args.dt, fps=args.fps, T=args.duration,
                                               damping=args.damping, ref_traj=args.ref_traj, Kp=args.Kp, Kv=args.Kv,
                                               a_ref=args.a_ref, f_ref=args.f_ref, with_cart=args.with_cart)
    if args.ref_traj:
        with open(controls_path, 'w') as f:
            json.dump(controls, f)
    if args.save_qs:
        qs_path = os.path.join(args.target_dir, 'q_{}.json'.format(i[0]))
        with open(qs_path, 'w') as f:
            json.dump(qs, f)


if __name__ == '__main__':
    seed = random.randint(0, 1000)
    pin.seed(seed)
    for i in tqdm(range(args.idx_start, args.idx_end)):
        l1 = np.random.uniform(args.min_bar_length, args.max_bar_length)
        bar_ratio = np.random.uniform(args.min_bar_ratio, args.max_bar_ratio)
        l = [l1, bar_ratio * l1]
        if args.N == 4:
            l = [l1, bar_ratio*l1, l1, bar_ratio*l1]
        mass_ratio = np.random.uniform(args.min_mass_ratio, args.max_mass_ratio)
        cart_mass = np.random.uniform(args.min_cart_mass, args.max_cart_mass)
        cart_length_radius_ratio = np.random.uniform(args.min_cart_length_radius_ratio, args.max_cart_length_radius_ratio)
        model, geom_model, visual_model = create_model(L=l, body_mass=args.body_mass, cart_mass=cart_mass, mass_ratio=mass_ratio, N=args.N,
                                                       with_cart=args.with_cart, cart_radius=args.cart_radius,
                                                       system_position=args.system_position,
                                                       cart_length_radius_ratio=cart_length_radius_ratio,
                                                       lower_limit=args.lower_limit, upper_limit=args.upper_limit)
        if args.N > 1:
            q = pin.randomConfiguration(model)
            q = np.array([q[i] for i in range(args.N)])
        elif args.N == 1:
            q = pin.randomConfiguration(model)
            q = pin.randomConfiguration(model)
        v = np.random.uniform(args.min_initial_v, args.max_initial_v, model.nv)
        config[i] = dict()
        config[i]['q'] = [x for x in q]
        config[i]['v'] = [x for x in v]
        config[i]['l'] = [x for x in l]
        if args.N > 1:
            config[i]['mass_ratio'] = mass_ratio
        if args.with_cart:
            p = Process(target=generate_sample, args=([i], q, v))
        else:
            p = Process(target=generate_sample, args=([i], q, v))
        p.start()
        p.join()

with open(os.path.join(args.target_dir, 'config_{}_{}.json'.format(args.idx_start, args.idx_end)), 'w') as f:
    json.dump(config, f)