# See all options
help() {
    python cifar10_cli.py --help
}

# Sanity check to run 1 iteration of train, test and val
dry_run() {
    python cifar10_cli.py \
        --default_root_dir $HOME/logs/ \
        --data_dir $HOME/data \
        --gpus 1 \
        --fast_dev_run 1
}

train() {
    python cifar10_cli.py \
        --default_root_dir $HOME/logs/ \
        --data_dir $HOME/data \
        --gpus 1 \
        --max_epochs 30
}

# help
# dry_run
train


