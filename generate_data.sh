mkdir "datasets/cartpole_128_128"
mkdir "datasets/cartpole_128_128/train"
mkdir "datasets/cartpole_128_128/test"
mkdir "datasets/cartpole_128_128/eval"
conda init bash
conda activate pino
echo "Activating pino env"


python odc/generate_simulations_pinocchio.py --idx_start 0 --idx_end 2 --target_dir datasets/cartpole_128_128/train/ \
--with_cart 1 --max_initial_v 0 --min_bar_length 0.4 --max_bar_length 0.9 --lower_limit 0 --upper_limit 4 \
--max_initial_v 0 --min_cart_length_radius_ratio 3 --max_cart_length_radius_ratio 7 --duration 5

python odc/generate_simulations_pinocchio.py --idx_start 0 --idx_end 2 --target_dir datasets/cartpole_128_128/test/ \
--with_cart 1 --max_initial_v 0 --min_bar_length 0.4 --max_bar_length 0.9 --lower_limit 0 --upper_limit 4 \
--max_initial_v 0 --min_cart_length_radius_ratio 3 --max_cart_length_radius_ratio 7 --duration 5

python odc/generate_simulations_pinocchio.py --idx_start 0 --idx_end 2 --target_dir datasets/cartpole_128_128/eval/ \
--with_cart 1 --max_initial_v 0 --min_bar_length 0.4 --max_bar_length 0.9 --lower_limit 0 --upper_limit 4 \
--max_initial_v 0 --min_cart_length_radius_ratio 3 --max_cart_length_radius_ratio 7 --duration 15

mkdir "datasets/cartpole_128_128_bw"
mkdir "datasets/cartpole_128_128_bw/train"
mkdir "datasets/cartpole_128_128_bw/test"
mkdir "datasets/cartpole_128_128_bw/eval"

python odc/dataset_utils/convert_to_bw.py --source_dir datasets/cartpole_128_128/train --target_dir datasets/cartpole_128_128_bw/train/ --out_channels 3
python odc/dataset_utils/convert_to_bw.py --source_dir datasets/cartpole_128_128/test --target_dir datasets/cartpole_128_128_bw/test/ --out_channels 3
python odc/dataset_utils/convert_to_bw.py --source_dir datasets/cartpole_128_128/eval --target_dir datasets/cartpole_128_128_bw/eval/ --out_channels 3


mkdir "datasets/cartpole_64_64_bw"

python odc/dataset_utils/resize_dataset.py --source_dir datasets/cartpole_128_128_bw  --target_dir datasets/cartpole_64_64_bw  --size 64

rm -rf datasets/cartpole_128_128_bw
rm -rf datasets/cartpole_128_128