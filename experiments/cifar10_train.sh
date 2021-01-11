# A series of improvement to the training for CIFAR 10 to train faster and
# better
#
# FunctionCall    Acc    Time   Model     Comments
# SimpleCNN       0.628  1m 53s SimpleCNN Conv2d, MaxPool, Relu
# DawnNetBaseline 0.831  8m 21s DawnNet   ResNet like skip layers

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
        --train_batch_size 128 \
        --fast_dev_run 1
}

# Baseline to start optimization
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
        --max_epochs 30
}

# DawnNet
train_DawnNetBaseline1() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --network ${NETWORK}\
        --default_root_dir $HOME/logs/${NETWORK}Baseline \
        --data_dir $HOME/data \
        --gpus 1 \
        --lr 1e-3 \
        --train_batch_size 128 \
        --max_epochs 30

#        --max_epochs 30
}

# help
# dry_run
# train_SimpleCNN
train_DawnNetBaseline


