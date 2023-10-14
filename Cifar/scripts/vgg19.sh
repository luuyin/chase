

cd ..



python3 main_dst_chase.py \
--stop_gmp_epochs  130 \
--minumum_ratio 0.2 \
--gpm_filter_pune \
--mest \
--mest_dst \
--filter_dst \
--mag_wise \
--layer_interval 8000 --start_layer_rate 0.5 \
--sparse \
--indicate_method test \
--model vgg19 --data cifar100 --decay-schedule cosine --seed 41 \
--sparse_init fixed_ERK --update_frequency 1000 --batch-size 100 --death-rate 0.05 \
--growth global_gradients --death global_magnitude  \
--redistribution none --epochs 160 --density 0.1 
