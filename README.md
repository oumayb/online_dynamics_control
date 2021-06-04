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