# Generate data
A pendulum dataset is provided as a zip file in ``odc/datasets/pendulum_v_low_eval_64_64_bw.zip``. 
To generate a cartpole dataset, run:
````
bash generate_data.sh
````
or edit it for other systems.

# Train model
````
python train.py --data_path datasets/simple_low --lr 1e-3 --ae_out_dim 8 --dataset_fraction 1 --delayed_dmd 1 \
--history 2 --a_method lagrangian --online_update 0 
````
Checkpoints will be saved in ``odc/runs/``

## Keep only models with lowest losses
````
bash clean_runs.sh
````

