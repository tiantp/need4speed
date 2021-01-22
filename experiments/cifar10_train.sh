# A series of improvement to the training for CIFAR 10 to train faster and
# achieve better accuracy
#
# Name           NAIVE BASE  CYC   SGD   AUG   ACS   ACS16  BSLR
# Optimize       Adam  Adam  SGD   SGD   Adam  SGD   SGD    SGD
# Epochs         35    35    35    35    35    35    35     30
# Batch Sz       128   128   128   128   128   128   128    512
# Use OneCycle   No    No    Yes   No    No    Yes   YES    YES
# FlipLR+Crop    No    No    No    No    Yes   Yes   YES    YES
# FP Precision   32    32    32    32    32    32    16     16
# Test Acc       63    83.3  78.7  67.7  89.6  91.5  93.5   92.3
# TotalTime (s)                                656   373    260

# See all options
help() {
    python cifar10_cli.py --help
}

# Sanity check to run 1 iteration of train, test and val
dry_run() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --network ${NETWORK}\
        --profiler simple \
        --default_root_dir $HOME/logs/dry_run \
        --data_dir $HOME/data \
        --gpus 1 \
        --lr 1e-3 \
        --lr_scheduler onecycle \
        --augment_data \
        --train_batch_size 128 \
        --fast_dev_run 1

}

# Small Network for Debugging purpose
# Around 63% test accuracy
train_NAIVE() {
    NETWORK=SimpleCNN
    python cifar10_cli.py \
        --network ${NETWORK} \
        --profiler simple \
        --default_root_dir $HOME/logs/${NETWORK}\
        --data_dir $HOME/data \
        --gpus 1 \
        --lr 1e-3 \
        --augment_data \
        --lr_scheduler onecycle \
        --optimizer_name sgd \
        --train_batch_size 128 \
        --max_epochs 35
}

# Baseline to start optimization
train_DawnNet_BASE() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --network ${NETWORK}\
        --profiler simple \
        --default_root_dir $HOME/logs/${NETWORK}Baseline \
        --data_dir $HOME/data \
        --gpus 1 \
        --lr 1e-3 \
        --lr_scheduler constant \
        --optimizer_name sgd \
        --train_batch_size 128 \
        --max_epochs 35
}

# Cyclic Learning rate
train_DawnNet_CYC() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --network ${NETWORK} \
        --profiler simple \
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
        --profiler simple \
        --default_root_dir $HOME/logs/${NETWORK}SGD\
        --data_dir $HOME/data \
        --gpus 1 \
        --lr_scheduler constant \
        --optimizer_name sgd \
        --train_batch_size 128 \
        --max_epochs 35
}

train_DawnNet_AUG() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --network ${NETWORK} \
        --profiler simple \
        --default_root_dir $HOME/logs/${NETWORK}AUG\
        --data_dir $HOME/data \
        --gpus 1 \
        --lr_scheduler constant \
        --optimizer_name adam \
        --train_batch_size 128 \
        --augment_data \
        --max_epochs 35
}

train_DawnNet_ACS() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --network ${NETWORK} \
        --profiler simple \
        --default_root_dir $HOME/logs/${NETWORK}ACS\
        --data_dir $HOME/data \
        --gpus 1 \
        --lr_scheduler onecycle \
        --optimizer_name sgd \
        --train_batch_size 128 \
        --augment_data \
        --max_epochs 35
}

# Switch to FP 16
train_DawnNet_ACS16() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --augment_data \
        --data_dir $HOME/data \
        --default_root_dir $HOME/logs/${NETWORK}ACS16\
        --gpus 1 \
        --lr_scheduler onecycle \
        --max_epochs 35 \
        --network ${NETWORK} \
        --optimizer_name sgd \
        --precision 16 \
        --profiler simple \
        --train_batch_size 128
}


# Increased Batch Size
train_DawnNet_BS512() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --augment_data \
        --data_dir $HOME/data \
        --default_root_dir $HOME/logs/${NETWORK}ACS16\
        --gpus 1 \
        --lr_scheduler onecycle \
        --max_epochs 33 \
        --network ${NETWORK} \
        --optimizer_name sgd \
        --precision 16 \
        --profiler simple \
        --train_batch_size 512
}

# Increased batch size, higher LR, reduced epochs
train_DawnNet_BSLR() {
    NETWORK=DawnNet
    python cifar10_cli.py \
        --augment_data \
        --data_dir $HOME/data \
        --default_root_dir $HOME/logs/${NETWORK}BSLR\
        --gpus 1 \
        --lr_scheduler onecycle \
        --max_epochs 30 \
        --network ${NETWORK} \
        --optimizer_name sgd \
        --oc_max_lr 0.44 \
        --precision 16 \
        --profiler simple \
        --train_batch_size 512
}

# Replace Batchnorm with Fixup init
train_Fixup() {
    NETWORK=Fixup
    python cifar10_cli.py \
        --augment_data \
        --data_dir $HOME/data \
        --default_root_dir $HOME/logs/${NETWORK}\
        --gpus 1 \
        --lr_scheduler onecycle \
        --max_epochs 30 \
        --network ${NETWORK} \
        --optimizer_name sgd \
        --oc_max_lr 0.15 \
        --precision 16 \
        --profiler simple \
        --train_batch_size 512
}


debug() {
    NETWORK=DawnNet
    # NETWORK=Fixup
    python cifar10_cli.py \
        --augment_data \
        --data_dir $HOME/data \
        --default_root_dir $HOME/logs/debug\
        --gpus 1 \
        --lr_scheduler onecycle \
        --max_epochs 2 \
        --network ${NETWORK} \
        --optimizer_name sgd \
        --oc_max_lr 0.05 \
        --precision 16 \
        --profiler simple \
        --train_batch_size 512
}





# help
# dry_run
# train_NAIVE
# train_DawnNet_Baseline
# train_DawnNet_CYC
# train_DawnNet_SGD
# train_DawnNet_AUG
# train_DawnNet_ACS
# train_DawnNet_ACS16
# train_DawnNet_BS512
# train_DawnNet_BSLR
# train_Fixup
debug





