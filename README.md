# Generate data
````
bash generate_data.sh
````

# Train model
````
python train.py --data_path datasets/simple_low --lr 1e-3 --ae_out_dim 8 --dataset_fraction 1 --delayed_dmd 1 \
--history 2 --a_method lagrangian --online_update 0 
````

## Keep only models with lowest losses
````
bash clean_runs.sh
````

