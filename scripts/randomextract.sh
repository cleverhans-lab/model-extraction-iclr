timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
DATASET='mnist'
architecture='MnistNetPate'
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python main.py \
--path /home/{USER}/capc-learning-main \
--data_dir /home/{USER}/data \
--dataset ${DATASET} \
--begin_id 0 \
--end_id 1 \
--num_querying_parties 1 \
--num_models 1 \
--num_epochs 50 \
--architecture ${architecture} \
--commands 'basic_model_stealing_attack' \
--class_type 'multiclass' \
--device_ids 0 1 2 3\
--momentum 0.5 \
--threshold 0 \
--sigma_gnmax 2.0 \
--sigma_gnmax_private_knn 28.0 \
--sigma_threshold 0 \
--lr 0.1 \
--weight_decay 1e-4 \
--scheduler_type 'ReduceLROnPlateau' \
--weak_classes '' \
--debug 'False' \
--query_set_type 'numpy' \
--pick_labels -1 \
--transfer_type '' \
--test_models_type 'private' \
--multilabel_prob_threshold 0.5 \
--log_every_epoch 0 \
--mode 'random' \
--num_optimization_loop 0 \
--adaptive_batch_size 1000 \
--target_model 'victim' \