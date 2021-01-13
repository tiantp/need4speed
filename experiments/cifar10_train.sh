# A series of improvement to the training for CIFAR 10 to train faster and
# better
#
#
# Name           SimpleCNN  Baseline  OneCycle SGD
# Optimize        Adam       Adam      Adam     SGD
# Use OneCycle    No         No        Yes      No
# FlipLR+Crop     No         No        No       No
# FP Precision    32         32        32       32
# Test Acc        63         83.3      78.7     67.7
# Time

# See all options
help() {
    python cifar10_cli.py --help
}

# Sanity check to run 1 iteration of train, test and val
dry_run() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --network ${NETWORK}\
        --default_root_dir $HOME/logs/dry_run \
        --data_dir $HOME/data \
        --gpus 1 \
        --lr 1e-3 \
        --lr_scheduler onecycle \
        --train_batch_size 128 \
        --fast_dev_run 1

}

# Small Network for Debugging purpose
# Around 63% test accuracy
train_SimpleCNN() {
    NETWORK=SimpleCNN
    python cifar10_cli.py \
        --network ${NETWORK} \
        --default_root_dir $HOME/logs/${NETWORK}\
        --data_dir $HOME/data \
        --gpus 1 \
        --max_epochs 1 \
        --lr 1e-3 \
        --train_batch_size 128 \
        --max_epochs 35
}

# Baseline to start optimization
train_DawnNet_Baseline() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --network ${NETWORK}\
        --default_root_dir $HOME/logs/${NETWORK}Baseline \
        --data_dir $HOME/data \
        --gpus 1 \
        --lr 1e-3 \
        --lr_scheduler constant \
        --train_batch_size 128 \
        --max_epochs 35
}

# Cyclic Learning rate
train_DawnNet_OneCycle() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --network ${NETWORK} \
        --default_root_dir $HOME/logs/${NETWORK}Onecycle\
        --data_dir $HOME/data \
        --gpus 1 \
        --lr_scheduler onecycle \
        --train_batch_size 128 \
        --max_epochs 35
}

train_DawnNet_SGD() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --network ${NETWORK} \
        --default_root_dir $HOME/logs/${NETWORK}SGD\
        --data_dir $HOME/data \
        --gpus 1 \
        --lr_scheduler constant \
        --optimizer_name sgd \
        --train_batch_size 128 \
        --max_epochs 35
}



# help
# dry_run
# train_SimpleCNN
# train_DawnNet_Baseline
# train_DawnNet_OneCycle
train_DawnNet_SGD


