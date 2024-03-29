 ## Code for paper [Online Learning and Control of Dynamical Systems from Sensory Input (NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/hash/ea6979872125d5acbac6068f186a0359-Abstract.html) 


# Files
The ```figures``` folder contains subfolders named after figures from the main submission and from the supplementary 
material. Each subfolder from ``fig1`` to ``fig5`` contains one GT video sequence, and predicted video sequences with 
the different models presented in the paper. 



# Code

## 0. Managing environments
For training, evaluation and control, use ```odc```.
````
cd online_dynamics_control
conda env create -f odc_environment.yml --name odc
conda activate odc
pip install -e .
````

An unactuated pendulum dataset can be downloaded from: https://mybox.inria.fr/f/27c2413966a844ea8506/?dl=1 .  
An actuated pendulum dataset can be downloaded from: https://mybox.inria.fr/f/564ebf586a8c45148202/?dl=1 .  
To generate more systems, see section 2.



## 1. Models
### 1.1 Train

First, make sure your dataset folder (the folder containing train and test folders) is in the ``datasets`` folder.

Then, run:
````
conda activate odc
cd odc
python train.py --data_path datasets/pendulum_v_low_eval_64_64_bw --lr 1e-3 --ae_out_dim 8 --dataset_fraction 1 \
--delayed_dmd 1 --history 2 --a_method prox --online_update 0 --exp_name test_run_
cd ..
````
Checkpoints will be saved in ``odc/runs/``

### 1.2 Keep only checkpoints with lowest loss
````
bash clean_runs.sh
````

### 1.3 Evaluate
````
cd odc
python evaluate_model.py --runs_dir models/ --exp_name simple_low_eval_prox_2021_5_23_21_54 \ 
--eval 1 --sample_duration 300 --metric_prefix sd=300_ --data_path datasets/pendulum_v_low_eval_64_64_bw
cd ..
````
Complete test_run_ with complete name with date

### 1.4 Control
This will generate Figure 7 of the supplementary material
````
cd odc
python control.py
cd ..
````
To generate Figure 6 from the main submission, run:
````
python odc/control.py --exp_path models/simple_l=0.6_control_sd=200_nptsA=150_2021_5_19_0_41 \ 
--model_path models/simple_l=0.6_control_sd=200_nptsA=150_2021_5_19_0_41/model_loss_8.262357005150989e-05.pth \ 
--data_path datasets/pendulum_v_l=0.6_control_sd=200_a=0.167_64_64_bw \ 
--system pendulum --video_idx 1301 --idx_init 151 --idx_final 178

````



# 2. Generate data
To generate a cartpole dataset, use the ```pino``` environment.
````
cd online_dynamics_control
conda env create -f pino_environment.yml --name pino
conda activate pino
pip install -e .
````
And run:
````
conda activate pino
bash generate_data.sh
````
or edit it for other systems.
