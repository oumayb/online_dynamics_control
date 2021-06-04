# Managing environments
````
cd online_dynamics_control
conda env create -f odc_environment.yml --name odc
conda activate odc
pip install -e .
````

A pendulum dataset is provided as a zip file in ``odc/datasets/pendulum_v_low_eval_64_64_bw.zip``. To generate more 
systems, see section 5.



# 1. Train model
````
python train.py --data_path datasets/simple_low --lr 1e-3 --ae_out_dim 8 --dataset_fraction 1 \
 --delayed_dmd 1 --history 2 --a_method prox --online_update 0 --exp_name test_run_
````
Checkpoints will be saved in ``odc/runs/``

## 1.1 Keep only models with lowest losses
````
bash clean_runs.sh
````

## 1.2 Evaluate model
````
python odc/evaluate_model.py --runs_dir odc/runs --exp_name test_run_ --eval 1 \ 
--sample_duration 300 --metric_prefix sd=300_
````
Complete test_run_ with complete name with date

## Control
This will generate Figure 7 of the supplementary material
````
python control.py
````
To generate Figure 6 from the main submission, run:
````
python control.py --exp_path ../../runs_dae/runs_dae/simple_l=0.6_control_sd=200_nptsA=150_2021_5_19_0_41 \ 
--model_path ../../runs_dae/runs_dae/simple_l=0.6_control_sd=200_nptsA=150_2021_5_19_0_41/model_loss_8.262357005150989e-05.pth \ 
--data_path ../../pendulum/datasets/pendulum_v_l=0.6_control_sd=200_a=0.167_64_64_bw \ 
--system pendulum --video_idx 1301 --idx_init 151 --idx_final 178

````



# 5. Generate data
````
cd online_dynamics_control
conda env create -f pino_environment.yml --name pino
conda activate pino
pip install -e .
````
To generate a cartpole dataset, run:
````
bash generate_data.sh
````
or edit it for other systems.